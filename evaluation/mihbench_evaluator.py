"""Evaluator for MIHBench yes/no tasks."""

from __future__ import annotations

from typing import Dict, Iterable, List


def compute_binary_metrics(preds: List[str], labels: List[str]) -> Dict[str, float]:
    p = [1 if x == "yes" else 0 for x in preds]
    y = [1 if x == "yes" else 0 for x in labels]

    tp = sum(int(pi == 1 and yi == 1) for pi, yi in zip(p, y))
    tn = sum(int(pi == 0 and yi == 0) for pi, yi in zip(p, y))
    fp = sum(int(pi == 1 and yi == 0) for pi, yi in zip(p, y))
    fn = sum(int(pi == 0 and yi == 1) for pi, yi in zip(p, y))

    total = max(len(y), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / total
    yes_ratio = sum(p) / total

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
