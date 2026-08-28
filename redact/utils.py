import os
import random

import numpy as np
import torch
import yaml
from scipy.stats import hmean


def log_print(*args, **kwargs):
    """Print that flushes immediately, bypassing conda run buffering.

    Drop-in replacement for ``print()`` — accepts the same arguments.
    Also writes to a log file if the ``LOG_FILE`` environment variable is set.
    """
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    log_path = os.environ.get("LOG_FILE")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            file_kwargs = {k: v for k, v in kwargs.items() if k != "flush"}
            file_kwargs["file"] = f
            print(*args, **file_kwargs)


def split_cot_answer(text: str) -> tuple[str, str]:
    """Split a generated reasoning response into ``(cot, answer)`` at ``</think>``.

    Always splits at the **first** ``</think>`` — whether or not it is
    followed by the canonical ``\\n\\n``.  After the split, up to two
    leading newlines are consumed from ``answer`` (so the canonical
    ``</think>\\n\\n`` separator behaves identically to a bare
    ``</think>``).  If ``</think>`` is not present at all, returns
    ``(text, "")``.

    Use this helper for *parsing* generated text into CoT and answer.
    Continue to use the canonical ``</think>\\n\\n`` (or
    :attr:`ReasoningGenerationEngine.THINK_SUFFIX`) when *constructing*
    model inputs that need to look like the training distribution.

    Why first-delim, not canonical-preferring
    -----------------------------------------
    Canonical-preferring split (the previous design) splits at the
    canonical ``</think>\\n\\n`` if one exists, which leaves any earlier
    bare ``</think>`` artifact inside the CoT.  Under heavy soft-token
    corruption the model regularly emits a stray ``</think>`` mid-CoT
    and then closes with a canonical ``</think>\\n\\n`` and nothing
    after; canonical-preferring put that stray ``</think>`` into
    ``cot`` and emitted an empty ``answer``, polluting downstream
    AFE/CFE evaluation.  First-delim is also how a human reader
    naturally parses these outputs.

    Examples
    --------
    >>> split_cot_answer("thinking</think>\\n\\nanswer")
    ('thinking', 'answer')
    >>> split_cot_answer("thinking</think>answer")
    ('thinking', 'answer')
    >>> split_cot_answer("first</think>mid</think>\\n\\nactual")
    ('first', 'mid</think>\\n\\nactual')
    >>> split_cot_answer("thinking, no close tag")
    ('thinking, no close tag', '')
    """
    delim = "</think>"
    if delim not in text:
        return text, ""
    cot, answer = text.split(delim, 1)
    # Consume up to two leading newlines so canonical ``</think>\n\n``
    # collapses to bare-``</think>`` semantics; further leading whitespace
    # is preserved.
    if answer.startswith("\n"):
        answer = answer[1:]
        if answer.startswith("\n"):
            answer = answer[1:]
    return cot, answer


def load_yaml(file_path):
    """Parse a YAML file and return its contents as a Python object.

    Thin wrapper over :func:`yaml.safe_load`; used to read the model
    and unlearning-config YAMLs. Returns whatever the document
    deserializes to (a ``dict`` for the config files in this repo).
    """
    with open(file_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_mu(results, subset_prefixes):
    """Compute Model Utility from evaluation results.

    MU = hmean over {rougeL_recall, cosine_similarity, entailment_score,
    token_entropy} evaluated on each subset prefix. With the canonical
    R-TOFU splits (retain, real_authors, world_facts) this is 12 values
    fed to a single hmean — matching the reference implementation at
    https://github.com/ssangyeon/R-TOFU (metrics/metrics.py), which
    per-split-per-metric means first, then hmeans.

    Caveat on cosine similarity: the R-TOFU paper text (EMNLP 2025,
    §5.1) describes CS as comparing pre- vs post-unlearning outputs,
    but the reference implementation computes CS gen-vs-gold — same as
    our existing ``CosineSimilarity`` evaluator. We follow the code,
    since that is what produced the paper's published numbers.

    Args:
        results: Dict of metric_key -> scalar mean, as produced by
            ``GenerationEngine.summary()``. Keys follow the
            ``{dataset}_{subset}_{metric}`` convention.
        subset_prefixes: Iterable of ``{dataset}_{subset}`` prefixes whose
            metrics should be aggregated. For the canonical MU this is
            three prefixes (retain, real_authors, world_facts).

    Returns:
        MU score (float). 0.0 if any required metric is missing or any
        component is non-positive (hmean is not defined on non-positive
        values). The failing key is logged via ``log_print``.
    """
    metrics = [
        "rougeL_recall",
        "cosine_similarity",
        "entailment_score",
        "token_entropy",
    ]
    scores = []
    missing = []
    for prefix in subset_prefixes:
        for metric in metrics:
            key = f"{prefix}_{metric}"
            if key not in results:
                missing.append(key)
            else:
                scores.append(results[key])
    if missing:
        log_print(
            f"[warn][compute_mu] {len(missing)} expected metric(s) missing; "
            f"first few: {missing[:4]}"
        )
        return 0.0
    nonpositive = [
        f"{prefix}_{metric}"
        for prefix in subset_prefixes
        for metric in metrics
        if results[f"{prefix}_{metric}"] <= 0
    ]
    if nonpositive:
        log_print(
            f"[warn][compute_mu] non-positive value(s) at {nonpositive[:3]}; MU=0"
        )
        return 0.0
    return float(hmean(scores))


def seed_everything(seed):
    """Seed every RNG this project draws from for reproducible runs.

    Covers the Python ``random`` module, NumPy's global RNG, the CPU
    Torch generator, and all CUDA device generators. Call once near the
    start of a script (or via :func:`enable_full_determinism`, which
    delegates the RNG seeding here and then layers on the backend
    determinism flags).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Seed every visible CUDA device's generator (no-op when CUDA is
    # unavailable), so multi-GPU runs are reproducible too.
    torch.cuda.manual_seed_all(seed)


def enable_full_determinism(seed: int, *, warn_only: bool = False) -> None:
    """Configure full bit-for-bit determinism for the current process.

    Calls :func:`seed_everything` for the standard RNG seeding, then
    layers on the PyTorch / CUDA knobs that ``seed_everything`` alone
    does not cover:

    - ``CUBLAS_WORKSPACE_CONFIG=:16:8`` so cuBLAS GEMMs (Linear /
      attention projections / the entire RoBERTa backbone) pick a
      reproducible workspace allocation. Must be set before the first
      cuBLAS handle is created; we set it here under the assumption
      that ``enable_full_determinism`` is called early in the script
      (no CUDA tensors materialized yet). The ``:16:8`` value matches
      both PyTorch and HuggingFace's documented defaults.
    - ``FLASH_ATTENTION_DETERMINISTIC=1`` covers the
      ``flash_attention_2`` backward path used by the reasoning-model
      and RegeneratingModel configs in this repo. No effect on the
      classifier path (RoBERTa-base uses eager / SDPA), but the helper
      lives in :mod:`redact.utils` and is share-usable across scripts.
    - ``torch.backends.cudnn.deterministic = True`` and
      ``torch.backends.cudnn.benchmark = False`` so cuDNN picks a
      single fixed algorithm per op rather than the fastest one for
      the current input shape (the benchmark path is the most common
      source of "same seed, different loss" complaints).
    - ``torch.use_deterministic_algorithms(True)`` so non-cuDNN ops
      (scatter, index_add, embedding backward) also pick their
      deterministic-but-slower implementations or raise loudly if no
      deterministic variant exists.

    Does NOT set ``CUDA_LAUNCH_BLOCKING=1``. That env var forces
    every kernel launch to be synchronous, costing 2-3× throughput
    for marginal additional determinism beyond what the knobs above
    already provide. Set it manually in the shell if you need full
    debug-mode synchronous CUDA (e.g. when chasing a determinism
    bug that the above flags don't catch).

    Parameters
    ----------
    seed
        RNG seed. Threaded through ``random``, ``numpy``, ``torch``,
        and ``torch.cuda`` via :func:`seed_everything`.
    warn_only
        Passed to :func:`torch.use_deterministic_algorithms`. When
        False, a non-deterministic op raises a ``RuntimeError`` at
        call time; when True (default), it warns and falls back to
        the non-deterministic implementation. The True default is
        the safer choice for long-running GPU jobs — a rarely-hit
        eval-time op without a deterministic implementation
        otherwise aborts a multi-hour run. Flip to False for strict
        debugging of irreproducibility sources.

    Raises
    ------
    Warns (via :func:`log_print`) when CUDA is already initialized
    at call time. ``CUBLAS_WORKSPACE_CONFIG`` only takes effect
    before the first cuBLAS handle; calling this function after a
    ``torch.zeros(1, device='cuda')`` (or any other CUDA tensor
    allocation) silently degrades the determinism contract. The
    warning catches the foot-gun where a refactor accidentally
    pushes CUDA initialization above the determinism setup.
    """
    if torch.cuda.is_initialized():
        log_print(
            "[warn][enable_full_determinism] CUDA was already "
            "initialized before this call. CUBLAS_WORKSPACE_CONFIG "
            "may have no effect on cuBLAS GEMMs in this process. "
            "Move the enable_full_determinism call earlier in the "
            "script (before any .cuda() / .to('cuda') / "
            "torch.cuda.* call) to restore the full determinism "
            "contract."
        )

    # CUBLAS_WORKSPACE_CONFIG must be set before any cuBLAS handle
    # exists in this process. Set it first.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"

    seed_everything(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


