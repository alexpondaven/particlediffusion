# Generate 1k images of "apple, white background, {adjective}" with 100 adjectives in csv (10 samples each)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import yaml
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import pandas as pd

# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None

# Set seed
seed=1
generator = torch.Generator("cuda").manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(0)
random.seed(0)

# Read config
with open("data/labels.yaml", 'r') as stream:
    labels = yaml.safe_load(stream)
with open("data/config.yaml", 'r') as stream:
    params = yaml.safe_load(stream)


adjectives = pd.read_csv("data/adjectives.csv",header=None)
for adj in adjectives[0]:
    # prompt = f"{label} on a white plate"
    prompt = f"apple, white background, {adj}"
    filename = "apple_" + adj.replace(" ", "_")
    dst_path = os.path.join("data","sample", "apple_adjectives")
    os.makedirs(dst_path, exist_ok=True)
    print(adj)
    for i in range(params['samples_per_class']):
        print(f"{filename}_{i}.png")
        img = pipe(prompt, 
                    num_inference_steps = params['num_steps'], 
                    guidance_scale = params['cfg'],
                    generator=generator).images[0]
        img.save(os.path.join(dst_path , f"{filename}_{i}.png"))