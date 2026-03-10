"""MUIRBench dataset loader (generic JSON/JSONL wrapper)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_muirbench(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
