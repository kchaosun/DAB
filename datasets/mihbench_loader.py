"""MIHBench dataset loader utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

TASK_TO_FILE = {
    "existence": "Questions/Existence/new_adv/coco_pope_adversarial_3.jsonl",
    "count": "Questions/Count/comparison_same_2.jsonl",
    "identity": "Questions/Id_Consitency/clip_questions_most_different_4.jsonl",
}


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_mihbench(dataset_root: str, task: str) -> List[Dict]:
    """Load MIHBench task split and normalize fields.

    Returns rows with keys: question_id, images, question, answer, task.
    """

    if task not in TASK_TO_FILE:
        raise ValueError(f"Unsupported task: {task}. choices={list(TASK_TO_FILE)}")

    path = str(Path(dataset_root) / TASK_TO_FILE[task])
    samples = _load_jsonl(path)
    normalized: List[Dict] = []
    for row in samples:
        normalized.append(
            {
                "question_id": row.get("question_id", row.get("id")),
                "images": row["image_list"],
                "question": row["text"],
                "answer": str(row["label"]).strip().lower(),
                "task": task,
                "meta": {k: v for k, v in row.items() if k not in {"image_list", "text", "label"}},
            }
        )
    return normalized
