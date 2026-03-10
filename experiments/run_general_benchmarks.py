"""General benchmark runner scaffold for MMIU / MUIRBench / MIRB."""

from __future__ import annotations

import argparse

from datasets.mirb_loader import load_mirb
from datasets.mmiu_loader import load_mmiu
from datasets.muirbench_loader import load_muirbench
from evaluation.general_evaluator import evaluate_yes_no_records


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["mmiu", "muirbench", "mirb"], required=True)
    p.add_argument("--data", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.benchmark == "mmiu":
        records = load_mmiu(args.data)
    elif args.benchmark == "muirbench":
        records = load_muirbench(args.data)
    else:
        records = load_mirb(args.data)

    if records and "prediction_text" in records[0] and "answer" in records[0]:
        print(evaluate_yes_no_records(records))
    else:
        print({"num_samples": len(records), "message": "Loaded dataset records."})


if __name__ == "__main__":
    main()
