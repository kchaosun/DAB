from transformers import AutoProcessor

from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from fdacd.fdacd_generate import fdacd_generate
from fdacd.utils_image_tokens import get_image_token_ranges_from_img_str_idx


# Minimal example (expects patched local modeling_qwen2_5_vl.py and processor inputs prepared externally).
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

# Assume `inputs` are prepared with processor(..., return_tensors="pt").
# Also assume `img_str_idx` was built by your DAB pre-processing.
# image_token_ranges = get_image_token_ranges_from_img_str_idx(img_str_idx)

# output_ids = fdacd_generate(
#     model,
#     input_ids=inputs["input_ids"].to(model.device),
#     attention_mask=inputs.get("attention_mask", None),
#     image_token_ranges=image_token_ranges,
#     gamma=0.3,
#     top_k=50,
#     alpha=0.3,
#     base_ratio=0.2,
#     img_str_idx=img_str_idx,
#     max_new_tokens=128,
#     pixel_values=inputs.get("pixel_values", None),
#     image_grid_thw=inputs.get("image_grid_thw", None),
# )
