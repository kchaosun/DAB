"""Unified generation method interface for baseline / DAB / FDACD."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from fdacd.fdacd_generate import fdacd_generate
from fdacd.utils_image_tokens import get_image_token_ranges_from_img_str_idx
from utils.image_utils import build_chat_messages, load_images


@torch.inference_mode()
def generate_with_method(
    model: Any,
    processor: Any,
    images: Sequence[str],
    question: str,
    method: str = "baseline",
    *,
    max_new_tokens: int = 128,
    alpha: float = 0.5,
    base_ratio: float = 0.2,
    gamma: float = 0.3,
    apc_top_k: int = 50,
    yes_no_prompt: bool = True,
    device: str = "cuda",
) -> str:
    """Run model generation using existing repo logic without re-implementing algorithms."""

    method = method.lower()
    prompt = question + (" Please answer with Yes or No." if yes_no_prompt else "")
    pil_images = load_images(images)
    messages = build_chat_messages(pil_images, prompt)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = _process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    img_str_idx = _build_img_str_idx(inputs["input_ids"][0], len(images))

    if method == "baseline":
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elif method == "dab":
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            img_str_idx=img_str_idx,
            alpha=alpha,
            base_ratio=base_ratio,
        )
    elif method == "fdacd":
        image_token_ranges = get_image_token_ranges_from_img_str_idx(img_str_idx)
        output_ids = fdacd_generate(
            model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            img_str_idx=img_str_idx,
            alpha=alpha,
            base_ratio=base_ratio,
            image_token_ranges=image_token_ranges,
            gamma=gamma,
            top_k=apc_top_k,
            max_new_tokens=max_new_tokens,
            eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
        )
    else:
        raise ValueError("method must be one of: baseline, dab, fdacd")

    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _process_vision_info(messages: List[Dict]):
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)
    except Exception:
        image_inputs = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        return image_inputs, None


def _build_img_str_idx(input_ids: torch.Tensor, num_images: int) -> List[float]:
    num_image_token = (input_ids == 151655).sum().item()
    img_start = (input_ids == 151652).nonzero(as_tuple=True)[0].tolist()
    per_img_token = num_image_token / max(num_images, 1)
    img_start.append(per_img_token)
    return img_start
