"""General evaluator entry points for non-MIHBench tasks."""

from __future__ import annotations

from typing import Dict, List

from evaluation.mihbench_evaluator import compute_binary_metrics
from utils.parsing import parse_yes_no


def evaluate_yes_no_records(records: List[Dict]) -> Dict[str, float]:
    preds = [parse_yes_no(r["prediction_text"]) for r in records]
    labels = [str(r["answer"]).lower() for r in records]
    return compute_binary_metrics(preds, labels)
