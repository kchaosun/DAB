from __future__ import annotations

from typing import Any


def model_supports_isolated_mode(model: Any) -> bool:
    """Best-effort check that the loaded model forward accepts FDACD args."""

    forward = getattr(model, "forward", None)
    if forward is None or not hasattr(forward, "__code__"):
        return False
    names = set(forward.__code__.co_varnames)
    return {"isolated_mode", "image_token_ranges"}.issubset(names)
