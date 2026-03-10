"""Run MIHBench experiments with baseline / DAB / FDACD methods."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from datasets.mihbench_loader import load_mihbench
from evaluation.mihbench_evaluator import compute_binary_metrics
from methods.method_registry import generate_with_method
from utils.image_utils import resolve_image_paths
from utils.parsing import parse_yes_no
from utils.result_logger import JsonlLogger, write_json
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen25vl", help="model alias name")
    p.add_argument("--model-path", required=True)
    p.add_argument("--method", choices=["baseline", "dab", "fdacd"], default="baseline")
    p.add_argument("--task", choices=["existence", "count", "identity"], required=True)
    p.add_argument("--dataset", required=True, help="repo root containing Questions/ and images")
    p.add_argument("--image-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--base-ratio", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--apc-top-k", type=int, default=50)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--local-rank", type=int, default=-1)
    return p.parse_args()


def load_qwen_model(model_path: str, device_map: str = "auto"):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quant_config = BitsAndBytesConfig(load_in_4bit=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model, processor


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    model, processor = load_qwen_model(args.model_path, device_map=args.device_map)

    samples = load_mihbench(args.dataset, args.task)
    logger = JsonlLogger(args.output)
    done_ids = logger.existing_ids()

    preds: List[str] = []
    labels: List[str] = []

    for sample in samples:
        qid = sample["question_id"]
        if qid in done_ids:
            continue

        image_paths = resolve_image_paths(args.image_root, sample["images"])
        pred_text = generate_with_method(
            model,
            processor,
            images=image_paths,
            question=sample["question"],
            method=args.method,
            max_new_tokens=args.max_new_tokens,
            alpha=args.alpha,
            base_ratio=args.base_ratio,
            gamma=args.gamma,
            apc_top_k=args.apc_top_k,
        )
        pred_label = parse_yes_no(pred_text)
        label = str(sample["answer"]).lower()

        logger.append(
            {
                "question_id": qid,
                "task": args.task,
                "method": args.method,
                "model": args.model,
                "question": sample["question"],
                "images": sample["images"],
                "answer": label,
                "prediction": pred_label,
                "prediction_text": pred_text,
            }
        )
        preds.append(pred_label)
        labels.append(label)

    # Metrics computed from full output file to include resumed runs.
    from utils.result_logger import load_jsonl

    records = [r for r in load_jsonl(args.output) if r.get("task") == args.task and r.get("method") == args.method]
    metrics = compute_binary_metrics([r["prediction"] for r in records], [r["answer"] for r in records])
    metrics.update({"task": args.task, "method": args.method, "model": args.model, "num_samples": len(records)})
    write_json(str(Path(args.output).with_suffix(".metrics.json")), metrics)
    print(metrics)


if __name__ == "__main__":
    main()
