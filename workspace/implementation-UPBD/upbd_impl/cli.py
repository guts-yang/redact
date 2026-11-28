"""CLI for isolated UPBD proposal implementations."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from upbd_impl.registry import get_spec, list_methods


def _train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train")
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--preset", choices=("smoke", "decisive"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--split-mode", choices=("entity", "question", "holdout"))
    parser.add_argument("--save-merged", action="store_true")


def _preset(args) -> dict:
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "presets.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(payload[args.preset])
    override = (payload.get("method_overrides", {})
                .get(args.method, {}).get(args.preset, {}))
    config.update(override)
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="upbd_impl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    _train_parser(sub)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--stage", choices=("gpu", "cpu"), required=True)
    evaluate.add_argument("--num-examples", type=int, default=4)
    args = parser.parse_args(argv)

    if args.command == "list":
        for spec in list_methods():
            print(f"{spec.slug}\t{spec.idea_dir}\t{spec.summary}")
        return 0

    if args.command == "train":
        spec = get_spec(args.method)
        preset = _preset(args)
        run_id = args.run_id or f"{spec.slug}_{args.preset}_s{args.seed}"
        config = dict(preset)
        config.update({
            "method": spec.slug,
            "run_id": run_id,
            "device": args.device,
            "preset": args.preset,
            "seed": args.seed,
            "save_merged": args.save_merged,
        })
        if args.steps is not None:
            config["steps"] = args.steps
        if args.max_len is not None:
            config["max_len"] = args.max_len
        if args.split_mode is not None:
            config["split_mode"] = args.split_mode
        if args.dry_run:
            print(json.dumps({
                "dry_run": True,
                "config": config,
                "results": os.environ.get("UPBD_IMPL_RESULTS", "<default:upbd_impl>"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
                "launched": False,
            }, ensure_ascii=False, indent=2))
            return 0
        from upbd_impl.runtime.trainer import TrainConfig, run_train
        run_train(TrainConfig(**config))
        return 0

    if args.command == "eval":
        from upbd_impl.runtime.rtofu_eval import run_eval
        run_eval(args.run_dir, stage=args.stage, num_examples=args.num_examples)
        return 0

    parser.error(f"unsupported command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
