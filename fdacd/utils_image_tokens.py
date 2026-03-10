from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def get_image_token_ranges_from_img_str_idx(img_str_idx: Sequence[float]) -> List[Tuple[int, int]]:
    """Convert DAB `img_str_idx` metadata into [start, end) image ranges.

    Expected format from existing DAB code:
      [img1_start, img2_start, ..., text_start, per_img_token]
    """

    if img_str_idx is None or len(img_str_idx) < 3:
        return []

    per_img_token = int(img_str_idx[-1])
    num_images = len(img_str_idx) - 2
    return [
        (int(img_str_idx[i]), int(img_str_idx[i]) + per_img_token)
        for i in range(num_images)
    ]


def get_image_token_ranges(
    input_ids: torch.LongTensor,
    image_start_token_id: int,
    image_token_id: int,
) -> List[Tuple[int, int]]:
    """Derive [start, end) image token spans from token ids for a single sample."""

    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    image_start_positions = (input_ids == image_start_token_id).nonzero(as_tuple=True)[0].tolist()
    if not image_start_positions:
        return []

    image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0].tolist()
    if not image_token_positions:
        return []

    ranges: List[Tuple[int, int]] = []
    pos_tensor = torch.tensor(image_token_positions, device=input_ids.device)
    for start in image_start_positions:
        following = pos_tensor[pos_tensor > start]
        if following.numel() == 0:
            continue
        block_start = int(following[0].item())

        # contiguous image-token run
        block_end = block_start
        while block_end < input_ids.numel() and int(input_ids[block_end].item()) == image_token_id:
            block_end += 1
        ranges.append((block_start, block_end))

    return ranges
