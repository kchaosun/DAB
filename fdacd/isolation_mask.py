from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def apply_cross_image_attention_mask(
    attn_scores: torch.Tensor,
    image_token_ranges: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Apply block-wise cross-image masking to attention scores.

    Any query token from image `i` attending key tokens from image `j` (i != j)
    is masked to -inf. The ranges are expected as [start, end) token spans.
    """

    if not image_token_ranges or len(image_token_ranges) <= 1:
        return attn_scores

    seq_len_q = attn_scores.shape[-2]
    seq_len_k = attn_scores.shape[-1]

    for i, (q_start, q_end) in enumerate(image_token_ranges):
        q_start = max(0, min(int(q_start), seq_len_q))
        q_end = max(q_start, min(int(q_end), seq_len_q))
        if q_start >= q_end:
            continue

        for j, (k_start, k_end) in enumerate(image_token_ranges):
            if i == j:
                continue
            k_start = max(0, min(int(k_start), seq_len_k))
            k_end = max(k_start, min(int(k_end), seq_len_k))
            if k_start >= k_end:
                continue
            attn_scores[..., q_start:q_end, k_start:k_end] = torch.finfo(attn_scores.dtype).min

    return attn_scores
