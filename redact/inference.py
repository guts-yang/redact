"""Inference / evaluation engines that drive batched generation.

The module exposes a small hierarchy of engines:

* :class:`InferenceEngine` — holds the shared wiring (model, tokenizer,
  data module, evaluator, batching) and the result-summarisation logic.
* :class:`GenerationEngine` — a generic free-form text generator that
  decodes completions, strips the echoed prompt, optionally length-matches
  gold/generated pairs, and scores them with one or more evaluators.
* :class:`ReasoningGenerationEngine` — the reasoning-model variant that
  splits each completion into a chain-of-thought and an answer at the
  ``</think>`` delimiter and reports AFE/CFE-style metrics separately.

``ReasoningGenerationEngine`` is the public entry point; the two bases are
internal and may be restructured freely as long as it (and the
``_remove_hooks`` helper, imported by the regenerating engine) keep their
observable behaviour.
"""

import time
from typing import Iterable

import numpy as np
from tqdm import tqdm

from redact.regen.model import RegeneratingModel
from redact.regen.utils import remove_hooks
from redact.utils import split_cot_answer


def _remove_hooks(model):
    """Detach any forward hooks currently registered on ``model``.

    ``model`` may be a bare model, an ``HFModel``-style wrapper, a
    ``RegeneratingModel`` (which tracks its own handles), or a
    ``ReasoningModel`` wrapping any of the above.  We unwrap one layer of
    ``ReasoningModel`` first, then dispatch: a ``RegeneratingModel`` cleans
    up through its handle-based method, a wrapper that exposes an inner
    ``.model`` has that submodule scrubbed, and anything else is scrubbed
    directly.
    """
    from redact.model.reasoning import ReasoningModel

    target = model._inner if isinstance(model, ReasoningModel) else model
    if isinstance(target, RegeneratingModel):
        target.remove_hooks()
    elif hasattr(target, "model"):
        remove_hooks(target.model)
    else:
        remove_hooks(target)


class InferenceEngine:
    """Base engine: stores the run configuration and aggregates results.

    Subclasses implement :meth:`inference`, populating ``self.results`` as a
    list of single-entry ``{key: values}`` dicts; :meth:`summary` collapses
    each entry to the mean of its value list.
    """

    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.subset_names = subset_names
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.prompt_prefix = prompt_prefix

    def prepare_dataset(self):
        """Load each requested subset into ``self.datasets`` (batched)."""
        self.datasets = {
            subset_name: self.data_module.load_dataset_for_eval(
                subset_name,
                load_in_batch=True,
                batch_size=self.batch_size,
                prompt_prefix=self.prompt_prefix,
            )
            for subset_name in self.subset_names
        }

    def inference(self):
        raise NotImplementedError(
            f"inference not implemented for {type(self).__name__}"
        )

    def summary(self):
        """Reduce ``self.results`` to per-key means.

        Each element of ``self.results`` is a one-item dict
        ``{key: value_list}``.  We emit ``{key: mean(value_list)}`` for
        every element, returning the list of those single-key mean dicts
        alongside the untouched original result dicts.
        """
        summary_stats, outputs = [], []
        for result in self.results:
            (key, value_list), = result.items()
            mean_stat = {key: float(np.mean(value_list))}
            summary_stats.append(mean_stat)
            print(mean_stat)
            outputs.append(result)
        return summary_stats, outputs


class GenerationEngine(InferenceEngine):
    """Generic text generator: decode completions, then score them.

    ``evaluator`` may be a single evaluator or a list; it is normalised to
    a list so multiple metrics can run over the same generations.
    """

    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
        comparison_length=128,
        truncate_answers=False,
    ):
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            evaluator,
            batch_size,
            prompt_prefix,
        )
        self.comparison_length = comparison_length
        self.truncate_answers = truncate_answers
        if not isinstance(self.evaluator, list):
            self.evaluator = [evaluator]

    def inference(self):
        """Generate completions for every subset and score them.

        Stores per-subset ``{"gold", "generated"}`` text under
        ``self.text_generations`` and appends one
        ``{dataset_subset_evaluator: scores}`` entry to ``self.results``
        for each evaluator.
        """
        self.results = []
        generations = self._generate()
        self.text_generations = {}
        for subset_name, data in generations.items():
            flat_gold = [item for batch in data["gold"] for item in batch]
            flat_generated = [
                item for batch in data["generated"] for item in batch
            ]
            self.text_generations[f"{self.data_module.name}_{subset_name}"] = {
                "gold": flat_gold,
                "generated": flat_generated,
            }
            for evaluator in self.evaluator:
                scores = []
                for _, gold, generated in tqdm(
                    zip(data["prompt"], data["gold"], data["generated"]),
                    total=len(data["gold"]),
                    desc=(
                        f"Evaluating {evaluator.name} of "
                        f"{self.data_module.name} on {subset_name}"
                    ),
                ):
                    scores.extend(evaluator.evaluate(gold, generated))
                self.results.append(
                    {
                        f"{self.data_module.name}_{subset_name}_{evaluator.name}": scores
                    }
                )

    def _generate(self):
        """Run batched generation, returning per-subset prompt/gold/generated
        lists (each a list-of-batches).  Forces left padding for generation
        and restores the original padding side afterwards.
        """
        self.prepare_dataset()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        generations = {}
        try:
            for subset_name, dataset in self.datasets.items():
                batched_prompts, batched_gold, batched_generated = [], [], []
                for batch in tqdm(
                    dataset,
                    desc=(
                        f"Generating completions of {self.data_module.name} "
                        f"on {subset_name}"
                    ),
                    total=len(dataset),
                ):
                    _remove_hooks(self.model)
                    prompts = batch[self.data_module.gen_prompt_key]
                    gold_answers = batch[self.data_module.gen_answer_key]

                    tokenized = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                    ).to(self.model.device)

                    generated = self.model.generate(
                        **tokenized,
                        prompts=prompts,
                        generation_config=self.model.generation_config,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    decoded = self.tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )

                    # Each decoded string still carries the echoed prompt;
                    # slice it off by character length.
                    completions = [
                        text[len(prompt):]
                        for prompt, text in zip(prompts, decoded)
                    ]
                    # Round-trip gold through the tokenizer so any special
                    # tokens it might contain are dropped consistently.
                    gold_answers = self.tokenizer.batch_decode(
                        self.tokenizer(
                            gold_answers, add_special_tokens=False
                        ).input_ids,
                        skip_special_tokens=True,
                    )
                    if self.truncate_answers:
                        gold_answers, completions = self.truncate(
                            gold_answers, completions
                        )

                    batched_prompts.append(prompts)
                    batched_gold.append(gold_answers)
                    batched_generated.append(completions)
                    _remove_hooks(self.model)

                assert (
                    len(batched_gold)
                    == len(batched_generated)
                    == len(batched_prompts)
                ), (
                    f"Length mismatch: {len(batched_gold)}, "
                    f"{len(batched_generated)}, {len(batched_prompts)}"
                )
                generations[subset_name] = {
                    "prompt": batched_prompts,
                    "gold": batched_gold,
                    "generated": batched_generated,
                }
        finally:
            self.tokenizer.padding_side = original_padding_side
        return generations

    def truncate(self, gold, generated):
        """Clip each gold/generated pair to their shared minimum length."""
        bounds = [min(len(g), len(h)) for g, h in zip(gold, generated)]
        clipped_gold = [g[:n] for g, n in zip(gold, bounds)]
        clipped_generated = [h[:n] for h, n in zip(generated, bounds)]
        return clipped_gold, clipped_generated


class ReasoningGenerationEngine(GenerationEngine):
    """GenerationEngine for reasoning models (e.g. DeepSeek-R1).

    Expects the model to be a ``ReasoningModel`` (or to expose
    ``n_think_tokens`` and prepend ``<think>\\n`` itself in ``generate()``).
    The engine does not insert think tokens — it only reads
    ``model.n_think_tokens`` so it can slice past the prefix when decoding.
    After generation it applies the BPE fix and splits each response at
    the ``</think>\\n\\n`` delimiter into CoT and answer portions.

    When the delimiter is missing — typically because ``max_new_tokens``
    truncated the generation before ``</think>`` could be emitted, or the
    output was otherwise malformed — the full response is kept as CoT and
    the answer is set to ``""``. There is no retry at a larger token
    budget; empty answers propagate to the ``apply_empty_override`` and
    ``think_completion_rate`` defenses used by AFE.

    After ``inference()`` completes, ``cot_generations`` and
    ``answer_generations`` are available as dicts keyed by
    ``{dataset_name}_{subset_name}``.
    """

    THINK_SUFFIX = "</think>\n\n"

    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        answer_evaluator,
        cot_evaluator=None,
        batch_size=64,
        prompt_prefix="",
        comparison_length=128,
        truncate_answers=False,
        apply_empty_override: "bool | Iterable[str]" = True,
    ):
        """See module docstring. ``apply_empty_override`` may be a bool
        (applies globally to every subset) or an iterable of subset names
        — the latter lets you enable the AFE empty-answer defense on the
        forget split while leaving utility splits (retain / real_authors
        / world_facts) unmodified, since on those splits an empty answer
        is a genuine utility failure, not an AFE gaming attempt.
        """
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            answer_evaluator,
            batch_size,
            prompt_prefix,
            comparison_length,
            truncate_answers,
        )
        self.cot_evaluator = cot_evaluator or []
        if not isinstance(self.cot_evaluator, list):
            self.cot_evaluator = [self.cot_evaluator]
        if isinstance(apply_empty_override, bool):
            self.apply_empty_override = apply_empty_override
        else:
            self.apply_empty_override = set(apply_empty_override)

    def _should_override_empty(self, subset_name):
        if isinstance(self.apply_empty_override, bool):
            return self.apply_empty_override
        return subset_name in self.apply_empty_override

    def _generate(self):
        self.prepare_dataset()
        padding_side = self.tokenizer.padding_side
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        n_think = getattr(self.model, "n_think_tokens", 0)

        subsets_generations = {}
        for subset_name, dataset in self.datasets.items():
            all_gold_answers, all_gold_cots, all_generated_answers = [], [], []
            all_generated_cot, all_generated_answer = [], []
            all_prompts = []
            total_time, total_examples = 0, 0

            for batch in tqdm(
                dataset,
                desc=f"Generating completions of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                _remove_hooks(self.model)
                prompts = batch[self.data_module.gen_prompt_key]
                gold_answers = batch[self.data_module.gen_answer_key]

                tokenized_prompts = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self.model.device)

                # The model (ReasoningModel) appends think tokens in generate(),
                # so decode_start accounts for both the tokenized prompt and
                # the think prefix that will be prepended.
                decode_start = tokenized_prompts["input_ids"].shape[1] + n_think

                start_time = time.perf_counter()
                generated = self.model.generate(
                    **tokenized_prompts,
                    prompts=prompts,
                    generation_config=self.model.generation_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                total_examples += len(prompts)

                # Decode only new tokens (after prompt + think prefix), apply BPE fix
                batch_responses = []
                for i in range(generated.shape[0]):
                    raw = self.tokenizer.decode(
                        generated[i][decode_start:], skip_special_tokens=True
                    )
                    batch_responses.append(raw)

                # Split into CoT and answer at </think> via split_cot_answer
                # (relaxed: matches </think> with any following whitespace,
                # which is more robust under corruption-induced perturbation
                # of the trailing newlines than the strict </think>\n\n).
                # If the delimiter is absent entirely (truncation, fully
                # incoherent output), the full response is treated as CoT
                # with an empty answer.
                batch_cot, batch_answer = [], []
                for resp in batch_responses:
                    cot, answer = split_cot_answer(resp)
                    batch_cot.append(cot)
                    batch_answer.append(answer)

                # Gold answers are plain text from the dataset — no special
                # tokens to strip.  Skipping the tokenize/decode roundtrip
                # avoids a lossy encode (this tokenizer merges "full name"
                # into a single "fullname" token, dropping the space).

                gold_cots = batch.get(getattr(self.data_module, "gen_cot_key", None), [""] * len(prompts))

                all_gold_answers.append(gold_answers)
                all_gold_cots.append(gold_cots)
                all_generated_answers.append(batch_responses)
                all_generated_cot.append(batch_cot)
                all_generated_answer.append(batch_answer)
                all_prompts.append(prompts)
                _remove_hooks(self.model)

            subsets_generations[subset_name] = {
                "prompt": all_prompts,
                "gold_answer": all_gold_answers,
                "gold_cot": all_gold_cots,
                "generated": all_generated_answers,
                "generated_cot": all_generated_cot,
                "generated_answer": all_generated_answer,
            }

        self.tokenizer.padding_side = padding_side
        return subsets_generations

    def inference(self):
        self.results = []
        answers = self._generate()
        self.text_generations = {}
        self.cot_generations = {}
        self.answer_generations = {}

        for subset_name, data in answers.items():
            key = f"{self.data_module.name}_{subset_name}"

            data_gold_answer = [item for sublist in data["gold_answer"] for item in sublist]
            data_gold_cot = [item for sublist in data["gold_cot"] for item in sublist]
            data_generated = [item for sublist in data["generated"] for item in sublist]
            data_cot = [item for sublist in data["generated_cot"] for item in sublist]
            data_answer = [item for sublist in data["generated_answer"] for item in sublist]

            self.text_generations[key] = {"gold": data_gold_answer, "generated": data_generated}
            self.cot_generations[key] = {"gold": data_gold_cot, "generated": data_cot}
            self.answer_generations[key] = {"gold": data_gold_answer, "generated": data_answer}

            # Run answer evaluators (AFE)
            for evaluator in self.evaluator:
                evaluator_outputs = []
                for gold, generated_answer in tqdm(
                    zip(data["gold_answer"], data["generated_answer"]),
                    total=len(data["gold_answer"]),
                    desc=f"Evaluating {evaluator.name} of {self.data_module.name} on {subset_name}",
                ):
                    outputs = evaluator.evaluate(gold, generated_answer)
                    evaluator_outputs.extend(outputs)
                self.results.append(
                    {f"{self.data_module.name}_{subset_name}_{evaluator.name}": evaluator_outputs}
                )

            # Override answer evaluator scores for empty-answer samples.
            # When </think> is missing, generated_answer is "". Evaluators
            # score empty-vs-gold as ~0.0, which yields 1-0=1.0 AFE —
            # rewarding garbage. Setting the raw score to 1.0 makes
            # 1-1.0=0.0, contributing AFE=0 for those samples.
            # Gated by apply_empty_override; pairs with (but is independent
            # of) the think_completion_rate this engine also emits.
            empty_mask = [ans == "" for ans in data_answer]
            if self._should_override_empty(subset_name) and any(empty_mask):
                for result_dict in self.results:
                    result_key = list(result_dict.keys())[0]
                    if result_key.startswith(key + "_") and "_cot_" not in result_key:
                        scores = result_dict[result_key]
                        for i, is_empty in enumerate(empty_mask):
                            if is_empty:
                                scores[i] = 1.0

            # Track think-completion rate for AFE penalty multiplier.
            n_total = len(data_answer)
            n_complete = sum(1 for ans in data_answer if ans != "")
            think_rate = n_complete / n_total if n_total > 0 else 0.0
            self.results.append({f"{key}_think_completion_rate": [think_rate]})

            # Run CoT evaluators (CFE). Skip for subsets that carry no gold
            # CoT (e.g. TOFU real_authors / world_facts, which use an empty
            # placeholder cot column so batchify works): stepwise evaluators
            # would score against "" and emit meaningless 0.0 metrics that
            # pollute saved results JSON.
            if not any(data_gold_cot):
                continue
            for evaluator in self.cot_evaluator:
                evaluator_outputs = []
                for gold_cot, generated_cot in tqdm(
                    zip(data["gold_cot"], data["generated_cot"]),
                    total=len(data["gold_cot"]),
                    desc=f"Evaluating {evaluator.name} (CoT) of {self.data_module.name} on {subset_name}",
                ):
                    outputs = evaluator.evaluate(gold_cot, generated_cot)
                    evaluator_outputs.extend(outputs)
                self.results.append(
                    {f"{self.data_module.name}_{subset_name}_cot_{evaluator.name}": evaluator_outputs}
                )
