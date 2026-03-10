"""Result logging helpers for per-sample prediction dumps and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class JsonlLogger:
    """Append-only JSONL logger with skip-finished support."""

    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def existing_ids(self, id_key: str = "question_id") -> set:
        if not self.output_path.exists():
            return set()
        ids = set()
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    ids.add(item[id_key])
                except Exception:
                    continue
        return ids

    def append(self, row: Dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
