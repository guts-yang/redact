import json
import os
import warnings

from datasets import Dataset, DatasetDict

from redact.dataset.base import BaseDataset


# Local JSONL vendoring of sangyon/R-TOFU github data/tofu/*.json
# (see data/rtofu/PROVENANCE.md for the pinned commit SHA). The HF dataset
# sangyon/R-TOFU drops the `task_id` field that the github JSONLs carry;
# we need task_id to build the paper's protocol retain splits.
_DATA_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "rtofu",
)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RTOFU(BaseDataset):
    dataset_type = "qa"
    path = "sangyon/R-TOFU"
    name = "rtofu"
    # Base splits loaded directly from data/rtofu/*.json. Forget splits are
    # filtered to the instance's task_id; retain splits are the "always-retain"
    # core across all task_ids in the matching forget series.
    forget_subsets = ["forget01", "forget05", "forget10"]
    retain_subsets = ["retain90", "retain50", "retain10"]
    # Protocol retain for forget`XX` task_id=K = retain(100 - n×XX) ∪ forgetXX[task_id≠K].
    # n = 10 for forget01/forget05, n = 9 for forget10 (9 task_ids vs 10).
    matched_retain_subsets = [
        "forget01_retain",
        "forget05_retain",
        "forget10_retain",
    ]
    # R-TOFU carries "Real Authors" and "World Facts" auxiliary QA splits in
    # its own repo as `{split}_perturbed.json`. Used only for Model Utility
    # (MU) evaluation. Loaded locally from data/rtofu/ (same pinned commit as
    # forget/retain) so R-TOFU's elaborated `original_answer` field is
    # available — the HuggingFace `locuslab/TOFU` aux configs carry only the
    # short bare answer ("William Shakespeare" vs R-TOFU's "The play 'Romeo
    # and Juliet' was written by William Shakespeare."), and the paper's MU
    # metric code explicitly keys off the elaborated string.
    tofu_aux_subsets = ["real_authors", "world_facts"]
    _tofu_aux_jsonl_suffix = "_perturbed.json"
    subsets = [
        "full",
        *retain_subsets,
        *forget_subsets,
        *matched_retain_subsets,
        *tofu_aux_subsets,
    ]
    # Callers access protocol retain via rtofu.dataset[match_retain[forget_split]].
    # The *name* is task_id-independent; the *content* depends on the instance's
    # task_id (set at __init__).
    match_retain = {
        "forget01": "forget01_retain",
        "forget05": "forget05_retain",
        "forget10": "forget10_retain",
    }
    keys = ["prompt", "answer", "prompt_formatted", "cot"]
    eval_prompt_key = "prompt_formatted"
    eval_answer_key = "answer"
    gen_prompt_key = "prompt_formatted"
    gen_answer_key = "answer"
    gen_cot_key = "cot"
    eval_dataset_keys = ["retain", "forget", "test"]

    # Canonical task_id for all forget splits. All experiments to date use
    # task_id=1 (matches the HF `sangyon/R-TOFU` forgetXX splits exactly);
    # other task_ids are available for sensitivity analysis but have not been
    # used in our pipeline.
    DEFAULT_TASK_ID = 1

    def __init__(
        self,
        formatting_tokens=None,
        eos_token=None,
        task_id=DEFAULT_TASK_ID,
        data_dir=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.formatting_tokens = formatting_tokens
        self.eos_token = eos_token if eos_token is not None else ""
        # The github JSONLs store `task_id` as a string (e.g. "1", "10"); we
        # accept int from Python callers and coerce with str() for matching.
        # Reject bool (isinstance(True, int) is True) and any non-int type --
        # silently coercing e.g. 1.0 -> "1.0" would leave the value unmatchable
        # against the JSONL's "1".
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise TypeError(
                f"task_id must be an int, got {type(task_id).__name__}: {task_id!r}"
            )
        self.task_id = str(task_id)
        self.data_dir = data_dir or _DATA_DIR_DEFAULT
        for k in ["prompt_prefix", "prompt_suffix", "answer_prefix", "answer_suffix"]:
            (
                setattr(self, k, formatting_tokens[k])
                if formatting_tokens is not None
                else setattr(self, k, "")
            )

    def download(self):
        """Load R-TOFU splits from local JSONLs and build protocol retain splits.

        `self.dataset` ends up as a DatasetDict with keys:
          - forget01/05/10: task_id-filtered to self.task_id
          - retain90/50/10: as-is (task_id-independent "always-retain" core)
          - forget01_retain / forget05_retain / forget10_retain: protocol retain
            for the instance's task_id, i.e. retain(100-n×XX) ∪ forgetXX[task_id≠K]
          - full: retain10 ∪ forget10.json (all task_ids) = 4000 rows, matching
            the HF `full` split exactly (3999 unique questions + one duplicate).

        forget01/05 carry task_ids 1..10; forget10 carries task_ids 1..9. If
        `self.task_id` is valid for some forget series but not others (e.g.
        task_id=10 exists for forget01/05 but not forget10), the invalid
        forget split and its matched retain are omitted from the DatasetDict
        so downstream access raises a clear KeyError rather than returning
        empty or misleading data.
        """
        raw = {}
        for name in self.forget_subsets + self.retain_subsets:
            path = os.path.join(self.data_dir, f"{name}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing R-TOFU data file: {path}. "
                    f"See data/rtofu/PROVENANCE.md for how to fetch."
                )
            raw[name] = _read_jsonl(path)

        available_task_ids = {
            name: sorted({row["task_id"] for row in raw[name]})
            for name in self.forget_subsets
        }
        valid_forget_subsets = [
            name for name in self.forget_subsets
            if self.task_id in available_task_ids[name]
        ]
        if not valid_forget_subsets:
            raise ValueError(
                f"task_id={self.task_id!r} is not valid for any forget split. "
                f"Available task_ids: "
                f"{ {n: tids for n, tids in available_task_ids.items()} }"
            )

        # Forget splits: filter by task_id, drop the task_id column since
        # downstream consumers expect the base schema (question/answer/cot).
        forget_filtered = {
            name: [
                {k: v for k, v in row.items() if k != "task_id"}
                for row in raw[name]
                if row["task_id"] == self.task_id
            ]
            for name in valid_forget_subsets
        }

        # Protocol retain: base retain ∪ forgetXX[task_id ≠ self.task_id]
        base_retain_for = {
            "forget01": "retain90",
            "forget05": "retain50",
            "forget10": "retain10",
        }
        matched_retain = {}
        for forget_name in valid_forget_subsets:
            base = raw[base_retain_for[forget_name]]
            unused = [
                {k: v for k, v in row.items() if k != "task_id"}
                for row in raw[forget_name]
                if row["task_id"] != self.task_id
            ]
            matched_retain[f"{forget_name}_retain"] = base + unused

        # Full: retain10 ∪ forget10 (all task_ids). Disjoint by construction
        # (retain10 = full − forget10[all]). 4000 rows total, 3999 unique.
        full_rows = raw["retain10"] + [
            {k: v for k, v in row.items() if k != "task_id"}
            for row in raw["forget10"]
        ]

        # Aux splits: vendored from R-TOFU's data/tofu/*_perturbed.json with
        # the same pinned commit. Optional — files may be absent in test
        # fixtures that only exercise forget/retain paths.
        aux_rows = {}
        for aux_name in self.tofu_aux_subsets:
            path = os.path.join(
                self.data_dir, f"{aux_name}{self._tofu_aux_jsonl_suffix}"
            )
            if not os.path.exists(path):
                continue
            # Rename `original_answer` -> `answer` so `gen_answer_key="answer"`
            # continues to work. The elaborated sentence is the string R-TOFU's
            # metric code uses as the gold for cosine and entailment on aux.
            # The short bare `answer` and `perturbed_answer` (wrong-option list)
            # aren't used in our pipeline, so we drop them here.
            aux_rows[aux_name] = [
                {
                    "question": row["question"],
                    "answer": row["original_answer"],
                    "cot": row.get("cot", ""),
                }
                for row in _read_jsonl(path)
            ]

        dataset_dict = {
            "full": Dataset.from_list(full_rows),
            **{n: Dataset.from_list(raw[n]) for n in self.retain_subsets},
            **{n: Dataset.from_list(forget_filtered[n]) for n in valid_forget_subsets},
            **{
                f"{n}_retain": Dataset.from_list(matched_retain[f"{n}_retain"])
                for n in valid_forget_subsets
            },
            **{n: Dataset.from_list(aux_rows[n]) for n in aux_rows},
        }
        self.dataset = DatasetDict(dataset_dict)

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix="",
        reasoned_idk_dir=None,
    ):
        if self.dataset is None:
            self.download()
        if split_name in self.tofu_aux_subsets and split_name not in self.dataset:
            raise KeyError(
                f"Aux split {split_name!r} not loaded. Expected vendored "
                f"JSONL at {os.path.join(self.data_dir, split_name)}"
                f"{self._tofu_aux_jsonl_suffix}; see data/rtofu/PROVENANCE.md."
            )
        dataset = self.dataset[split_name]
        dataset = dataset.rename_column("question", "prompt")
        # Defensive: production-path aux splits carry `cot` from the vendored
        # JSONL's always-empty `cot` field. Tests that inject a partial
        # fixture may omit it; batchify requires every key in self.keys.
        if "cot" not in dataset.column_names:
            dataset = dataset.add_column("cot", [""] * len(dataset))
        dataset = dataset.map(
            lambda x: {
                "prompt_formatted": f"{self.prompt_prefix}{x['prompt']}{self.prompt_suffix}",
                "answer": self.answer_prefix + x["answer"] + self.eos_token,
            }
        )
        dataset = dataset.map(
            lambda x: {"prompt_formatted": prompt_prefix + x["prompt_formatted"]}
        )
        if reasoned_idk_dir is not None:
            dataset = self._augment_with_reasoned_idk(
                dataset, split_name, reasoned_idk_dir
            )
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def _augment_with_reasoned_idk(self, dataset, split_name, data_dir):
        """Merge per-example reasoned IDK data from a local JSON file.

        Adds ``reasoned_idk_cot`` and ``reasoned_idk_answer`` columns when
        ``{data_dir}/reasoned_idk_{split_name}.json`` exists and its length
        matches the dataset.  Returns the dataset unchanged otherwise.
        """
        path = os.path.join(data_dir, f"reasoned_idk_{split_name}.json")
        if not os.path.exists(path):
            return dataset
        with open(path) as f:
            idk_entries = json.load(f)
        if len(idk_entries) != len(dataset):
            warnings.warn(
                f"Reasoned IDK length mismatch: {len(idk_entries)} entries "
                f"vs {len(dataset)} dataset rows for split '{split_name}'. "
                f"Skipping augmentation."
            )
            return dataset
        # Validate positional alignment via idx field
        for j, entry in enumerate(idk_entries):
            if entry.get("idx") != j:
                warnings.warn(
                    f"Reasoned IDK idx mismatch at position {j}: "
                    f"expected idx={j}, got idx={entry.get('idx')}. "
                    f"Skipping augmentation."
                )
                return dataset
        dataset = dataset.add_column(
            "reasoned_idk_cot", [e["reasoned_idk_cot"] for e in idk_entries]
        )
        dataset = dataset.add_column(
            "reasoned_idk_answer", [e["reasoned_idk_answer"] for e in idk_entries]
        )
        return dataset

    def load_dataset_for_classification(self, split_name, use_val=False):
        if self.dataset is None:
            self.download()
        assert (
            split_name in self.forget_subsets and split_name in self.match_retain
        ), f"Invalid split name: {split_name}"
        retain_set_name = self.match_retain[split_name]
        forget_set_name = split_name

        retain_dataset = self.dataset[retain_set_name]
        forget_dataset = self.dataset[forget_set_name]
        full_dataset = self.dataset["full"]

        retain_dataset, forget_dataset, full_dataset = map(
            lambda x: x.rename_column("question", "text").remove_columns(
                [c for c in ["answer", "cot"] if c in x.column_names]
            ),
            [retain_dataset, forget_dataset, full_dataset],
        )

        retain_dataset = retain_dataset.map(lambda x: {"label": 0})
        forget_dataset = forget_dataset.map(lambda x: {"label": 1})
        train_dataset = Dataset.from_dict(
            {
                "text": list(retain_dataset["text"]) + list(forget_dataset["text"]),
                "label": list(retain_dataset["label"]) + list(forget_dataset["label"]),
            }
        )
        if use_val:
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset, val_dataset = split["train"], split["test"]

        general_dataset = full_dataset.map(lambda x: {"label": 0})

        dataset = DatasetDict(
            {
                "train": train_dataset,
                "retain": retain_dataset,
                "forget": forget_dataset,
                "test": general_dataset,
            }
        )
        if use_val:
            dataset["valid"] = val_dataset
        return dataset
