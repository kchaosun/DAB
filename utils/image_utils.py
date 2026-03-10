"""Image and multimodal input preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


def load_images(image_paths: Sequence[str], image_size: Tuple[int, int] | None = (24 * 28, 24 * 28)) -> List[Image.Image]:
    """Load and optionally resize images for multimodal prompting."""
    images: List[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        if image_size is not None:
            img = img.resize(image_size)
        images.append(img)
    return images


def build_chat_messages(images: Sequence[Image.Image], question: str) -> List[Dict]:
    """Build Qwen-VL style chat message payload."""
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def resolve_image_paths(image_root: str, relative_paths: Sequence[str]) -> List[str]:
    """Resolve sample image paths against root folder."""
    root = Path(image_root)
    return [str(root / p) for p in relative_paths]
