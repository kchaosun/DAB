from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _apc_topk_blend(
    logits_base: torch.Tensor,
    logits_iso: torch.Tensor,
    gamma: float,
    top_k: int,
) -> torch.Tensor:
    fdacd_logits = logits_base.clone()
    v = logits_base.shape[-1]
    k = min(max(int(top_k), 1), v)

    topk_idx = torch.topk(logits_base, k=k, dim=-1).indices
    blended = logits_base + gamma * (logits_iso - logits_base)
    fdacd_logits.scatter_(dim=-1, index=topk_idx, src=blended.gather(-1, topk_idx))
    return fdacd_logits


def _sample_next_token(logits: torch.Tensor, temperature: float = 0.0) -> torch.LongTensor:
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def fdacd_generate(
    model: Any,
    *,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    image_token_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    max_new_tokens: int = 128,
    gamma: float = 0.3,
    top_k: int = 50,
    temperature: float = 0.0,
    alpha: float = 0.0,
    base_ratio: float = 0.2,
    img_str_idx: Optional[Sequence[float]] = None,
    eos_token_id: Optional[int] = None,
    **model_kwargs: Dict[str, Any],
) -> torch.LongTensor:
    """Training-free FDACD decoding with two forwards per token step."""

    device = input_ids.device
    generated = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(generated, device=device)

    common_kwargs = dict(model_kwargs)
    common_kwargs.update(
        {
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "output_hidden_states": False,
        }
    )

    base_cache = None
    iso_cache = None

    for step in range(max_new_tokens):
        step_input_ids = generated if step == 0 else generated[:, -1:]
        step_attention_mask = attention_mask if step == 0 else attention_mask[:, -1:]

        out_base = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            past_key_values=base_cache,
            img_str_idx=img_str_idx,
            alpha=alpha,
            base_ratio=base_ratio,
            isolated_mode=False,
            image_token_ranges=image_token_ranges,
            **common_kwargs,
        )
        base_cache = out_base.past_key_values

        out_iso = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            past_key_values=iso_cache,
            img_str_idx=img_str_idx,
            alpha=alpha,
            base_ratio=base_ratio,
            isolated_mode=True,
            image_token_ranges=image_token_ranges,
            **common_kwargs,
        )
        iso_cache = out_iso.past_key_values

        logits_base = out_base.logits[:, -1, :]
        logits_iso = out_iso.logits[:, -1, :]

        logits_fdacd = _apc_topk_blend(logits_base, logits_iso, gamma=gamma, top_k=top_k)
        next_token = _sample_next_token(logits_fdacd, temperature=temperature)

        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token, device=device)], dim=-1)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

    return generated
