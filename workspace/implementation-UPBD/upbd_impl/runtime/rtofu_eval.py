"""Run-local adapters for existing R-TOFU generation and Hy4 evaluation CLIs."""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from upbd_impl.runtime.artifacts import RunArtifacts
from upbd_impl.runtime.data import normalize_question, read_csv_rows
from upbd_impl.runtime.paths import Paths


SPLITS = ("forget10", "forget10_retain", "real_authors", "world_facts")
SPLIT_CHANNELS = {
    "forget10": "forget",
    "forget10_retain": "retain",
    "real_authors": "authors",
    "world_facts": "world",
}


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _model_config(artifacts: RunArtifacts, paths: Paths) -> Path:
    merged = artifacts.path("merged")
    if not (merged / "config.json").is_file():
        raise FileNotFoundError(
            "evaluation requires a merged checkpoint; train with --save-merged first")
    import yaml

    payload = {
        "embedding_dim": 4096,
        "embedding_module": "model.embed_tokens",
        "formatting_tokens": {
            "answer_prefix": "", "answer_suffix": "",
            "prompt_prefix": "<｜User｜>", "prompt_suffix": "<｜Assistant｜>",
        },
        "hf_name": str(merged),
        "tokenizer_name": str(paths.target_model),
        "attn_implementation": "sdpa",
        "torch_dtype": "float32",
        "load_in_8bit": False,
        "load_in_4bit": False,
        "baseline": True,
    }
    target = artifacts.write_text("model_config/upbd_impl.yaml", yaml.safe_dump(payload, sort_keys=False))
    return target.parent


def generation_commands(artifacts: RunArtifacts, paths: Paths,
                        config_dir: Path, num_examples: int) -> list[list[str]]:
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")
    return [
        [
            sys.executable, "-m", "scripts.generate_rtofu_responses",
            "--model_name", "upbd_impl",
            "--model_config_dir", str(config_dir),
            "--split", split,
            "--output", str(artifacts.path(f"eval/raw/{split}.csv")),
            "--max_new_tokens", "1024",
            "--batch_size", "4",
            "--num_examples", str(num_examples),
        ]
        for split in SPLITS
    ]


def select_test_csvs(artifacts: RunArtifacts) -> dict[str, int]:
    """Filter raw generated rows to the run manifest's held-out test partition."""
    manifest_path = artifacts.path("split_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("evaluation requires the run's split_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for split in SPLITS:
        channel = SPLIT_CHANNELS[split]
        entries = manifest["channels"][channel]["test"]
        questions = {normalize_question(entry.get("question") or "")
                     for entry in entries if entry.get("question")}
        indices = {str(entry["probe_id"]) for entry in entries if "probe_id" in entry}
        raw_path = artifacts.path(f"eval/raw/{split}.csv")
        rows = read_csv_rows(raw_path)
        selected = [
            row for row in rows
            if (normalize_question(row.get("question") or "") in questions
                or str(row.get("index")) in indices)
        ]
        if not selected:
            raise RuntimeError(
                f"no generated rows belong to the {split} test partition; "
                "run full generation or ensure the smoke sample covers test rows")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
        artifacts.write_text(f"eval/test/{split}.csv", buffer.getvalue())
        counts[split] = len(selected)
    artifacts.write_json("eval/test_selection.json", {
        "mode": manifest.get("mode"),
        "scientific_valid": manifest.get("scientific_valid", False),
        "counts": counts,
    })
    return counts


def _judge_commands(artifacts: RunArtifacts, workers: int, timeout: int) -> list[list[str]]:
    csv_path = str(artifacts.path("eval/test/forget10.csv"))
    afe = str(artifacts.path("eval/test/afe.hy4.jsonl"))
    cfe = str(artifacts.path("eval/test/cfe.hy4.jsonl"))
    return [
        [
            sys.executable, "-m", "scripts.hy4_judge_afe_rtofu",
            "--csv", csv_path, "--split", "forget10", "--output_jsonl", afe,
            "--workers", str(workers), "--max_retries", "2", "--timeout", str(timeout),
        ],
        [
            sys.executable, "-m", "scripts.hy4_judge_cfe_rtofu",
            "--csv", csv_path, "--split", "forget10", "--output_jsonl", cfe,
            "--workers", str(workers), "--max_retries", "2", "--timeout", str(timeout),
        ],
        [
            sys.executable, "-m", "scripts.ensure_hy4_judges",
            "--csv", csv_path, "--split", "forget10",
            "--afe", afe, "--cfe", cfe,
            "--rounds", "3", "--workers", str(workers), "--timeout", str(timeout),
        ],
    ]


def _aggregate_command(artifacts: RunArtifacts) -> list[str]:
    return [
        sys.executable, "-m", "scripts.evaluate_rtofu_from_csv",
        "--csv", str(artifacts.path("eval/test/forget10.csv")),
        "--split", "forget10",
        "--judge_afe_scores", str(artifacts.path("eval/test/afe.hy4.jsonl")),
        "--rtofu_judge_scores", str(artifacts.path("eval/test/cfe.hy4.jsonl")),
        "--mu_retain_csv", str(artifacts.path("eval/test/forget10_retain.csv")),
        "--mu_real_authors_csv", str(artifacts.path("eval/test/real_authors.csv")),
        "--mu_world_facts_csv", str(artifacts.path("eval/test/world_facts.csv")),
        "--output_json", str(artifacts.path("eval/test/eval.json")),
    ]


def run_eval(run_dir: str | Path, *, stage: str, num_examples: int = 4) -> None:
    artifacts = RunArtifacts.existing(run_dir)
    paths = Paths.from_env()
    env = os.environ.copy()
    # The project judge is fixed to Hy4; no alternate judge is selected here.
    env["HY4_MODEL"] = "hy4"
    env["LLM_MODEL"] = "hy4"

    if stage == "gpu":
        config_dir = _model_config(artifacts, paths)
        from upbd_impl.runtime.device import resolve_device
        resolve_device("cuda")
        for split in SPLITS:
            artifacts.path(f"eval/raw/{split}.csv").parent.mkdir(parents=True, exist_ok=True)
        for command in generation_commands(artifacts, paths, config_dir, num_examples):
            _run(command, cwd=paths.redact_root, env=env)
        for split in SPLITS:
            path = artifacts.path(f"eval/raw/{split}.csv")
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"generation did not produce {path}")
        artifacts.write_json("eval_gpu_complete.json", {
            "complete": True, "splits": list(SPLITS), "num_examples": num_examples,
            "note": "raw generation only; CPU stage selects manifest test rows",
        })
        return

    if stage != "cpu":
        raise ValueError("stage must be gpu or cpu")
    if not artifacts.path("eval_gpu_complete.json").is_file():
        raise RuntimeError("GPU generation stage has not completed")
    counts = select_test_csvs(artifacts)
    env["CUDA_VISIBLE_DEVICES"] = ""
    workers = int(os.environ.get("UPBD_IMPL_HY4_WORKERS", "2"))
    timeout = int(os.environ.get("UPBD_IMPL_HY4_TIMEOUT", "180"))
    for command in _judge_commands(artifacts, workers, timeout):
        _run(command, cwd=paths.redact_root, env=env)
    _run(_aggregate_command(artifacts), cwd=paths.redact_root, env=env)
    output = artifacts.path("eval/test/eval.json")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("evaluation aggregate did not produce eval.json")
    artifacts.write_json("eval_cpu_complete.json", {
        "complete": True, "judge": "hy4", "eval": str(output), "test_counts": counts,
    })
