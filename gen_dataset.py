## Generate most diverse set of images

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import yaml
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
print("CUDA?: ", torch.cuda.is_available())

# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None

# Set seed
seed=1024
generator = torch.Generator("cuda").manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(0)
random.seed(0)

# Read config
with open("data/labels.yaml", 'r') as stream:
    labels = yaml.safe_load(stream)
with open("data/config.yaml", 'r') as stream:
    params = yaml.safe_load(stream)

# Sample from dataset classes
dataset = 'fruit_white' ## Specify dataset (classes specified in data/labels.yaml)
for label in labels[dataset]:
    # prompt = f"{label} on a white plate"
    prompt = f"{label}, white background"
    filename = label.replace(" ", "_")
    dst_path = os.path.join("data",dataset, filename)
    os.makedirs(dst_path, exist_ok=True)
    print(label)
    for i in range(params['samples_per_class']):
        print(f"{filename}_{i}.png")
        img = pipe(prompt, 
                    num_inference_steps = params['num_steps'], 
                    guidance_scale = params['cfg'],
                    generator=generator).images[0]
        img.save(os.path.join(dst_path , f"{filename}_{i}.png"))