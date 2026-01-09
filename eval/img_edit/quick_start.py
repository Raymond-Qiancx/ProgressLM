import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("/projects/p32958/chengxuan/models/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images/h5_agilex_3rgb/37_putegg/2024_10_14-10_58_35-172891272087527168.00/camera_front_0128.jpg").convert("RGB")
prompt = "Replace the blue egg with a green apple in the robotic gripper."


inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))