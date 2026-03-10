#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs}"
CSV_PATH="$OUT_DIR/results.csv"
MD_PATH="$OUT_DIR/results.md"

python - "$OUT_DIR" "$CSV_PATH" "$MD_PATH" <<'PY'
import csv
import glob
import json
import os
import sys
from collections import defaultdict

out_dir, csv_path, md_path = sys.argv[1:4]
metric_files = sorted(glob.glob(os.path.join(out_dir, "*.metrics.json")))
rows = []
for p in metric_files:
    with open(p, "r", encoding="utf-8") as f:
        rows.append(json.load(f))

fieldnames = ["model", "method", "task", "accuracy", "precision", "recall", "f1", "yes_ratio", "num_samples"]
os.makedirs(out_dir, exist_ok=True)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})

pivot = defaultdict(dict)
for r in rows:
    pivot[(r.get("model"), r.get("method"))][r.get("task")] = r.get("accuracy", 0.0)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("| Model | Method | Existence | Count | Identity |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for (model, method), task_map in sorted(pivot.items()):
        f.write(
            f"| {model} | {method} | {task_map.get('existence', 0):.4f} | {task_map.get('count', 0):.4f} | {task_map.get('identity', 0):.4f} |\\n"
        )
print(f"Wrote {csv_path} and {md_path}")
PY
