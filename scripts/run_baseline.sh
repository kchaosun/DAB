#!/usr/bin/env bash
set -euo pipefail

python experiments/run_mihbench.py \
  --model qwen25vl \
  --model-path "${MODEL_PATH:?set MODEL_PATH}" \
  --method baseline \
  --task "${TASK:-existence}" \
  --dataset "${DATASET_ROOT:-.}" \
  --image-root "${IMAGE_ROOT:?set IMAGE_ROOT}" \
  --output "outputs/qwen25vl_baseline_${TASK:-existence}.jsonl"
