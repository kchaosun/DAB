from __future__ import annotations

from typing import Any, Dict, Optional, Sequence


def inject_dab_kwargs(
    model_inputs: Dict[str, Any],
    *,
    img_str_idx: Optional[Sequence[float]],
    alpha: float,
    base_ratio: float,
) -> Dict[str, Any]:
    """Attach DAB kwargs to a model forward/generate call."""

    model_inputs = dict(model_inputs)
    model_inputs["img_str_idx"] = img_str_idx
    model_inputs["alpha"] = alpha
    model_inputs["base_ratio"] = base_ratio
    return model_inputs
