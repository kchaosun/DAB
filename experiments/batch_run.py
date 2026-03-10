"""Batch launcher for model x method x MIHBench-task sweeps."""

from __future__ import annotations

import argparse
import itertools
import subprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="format name=path")
    p.add_argument("--methods", nargs="+", default=["baseline", "dab", "fdacd"])
    p.add_argument("--tasks", nargs="+", default=["existence", "count", "identity"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_pairs = [m.split("=", 1) for m in args.models]

    for (model_name, model_path), method, task in itertools.product(model_pairs, args.methods, args.tasks):
        out = f"{args.output_dir}/{model_name}_{method}_{task}.jsonl"
        cmd = [
            "python",
            "experiments/run_mihbench.py",
            "--model",
            model_name,
            "--model-path",
            model_path,
            "--method",
            method,
            "--task",
            task,
            "--dataset",
            args.dataset,
            "--image-root",
            args.image_root,
            "--output",
            out,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
