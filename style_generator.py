import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
from diffusers import DiffusionPipeline
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import glob
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import pandas as pd

from fid.fid_score import calculate_activation_statistics, save_statistics, load_statistics, calculate_frechet_distance, get_activations
from fid.inception import InceptionV3

from PIL import Image
from tqdm.auto import tqdm
from torch import autocast

from src.visualise import image_grid, latent_to_img, decode_latent, output_to_img
from schedulers.euler_discrete import EulerDiscreteCustomScheduler, FrozenDict, randn_tensor
from src.score_utils import get_sigmas, get_score_input, scale_input, get_score, step_score, denoise
from src.sampling_utils import random_step, langevin_step, repulsive_step_parallel
from src.kernel import RBF
from src.embedding import CNN64, CNN16, init_weights, AverageDim, Average, VAEAverage, Style

import torch
import torch.nn as nn
import torch.nn.functional as F
# import autograd.numpy as anp
from collections import deque

# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
# model_id = "CompVis/stable-diffusion-v1-4"

# tf32 faster computation with Ampere
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda"
dtype=torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
# pipe = pipe.to(device)
pipe.safety_checker = None
pipe.enable_attention_slicing() # NOTE: 10% slower inference, but big memory savings
# pipe.enable_sequential_cpu_offload() # NOTE: May slow down inference a lot
pipe.enable_vae_slicing() # TODO: Try to give batches to VAE
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# Settings
config = {
    "pipe": pipe,
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "num_train_timesteps": 1000,
    "num_init_latents": 1,
    "batch_size": 1,
    "cfg": 8,
    "beta_start": 0.00085,
    "beta_end": 0.012,
}

# Seeds
seed=0
generator = torch.Generator(device).manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(0)
random.seed(0)

# Noise levels
sigmas, timesteps = get_sigmas(config, device=device)
config = {**config,
          "sigmas": sigmas,
          "timesteps": timesteps,
          }



# For each style, create 500 images of "tree, {style}"
# Store all 21 latents as a torch
styles =  pd.read_csv("data/styles/styles.csv",header=None)
for style in styles[0]:
    prompt = f"tree, {style}"
    filename = "tree_" + style.replace(" ", "_")
    dst_path = os.path.join("data","styles", style.replace(" ", "_"))
    os.makedirs(dst_path, exist_ok=True)
    print(style)
    for i in range(1000):
        init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device=device, dtype=dtype)
        config = {**config,
                "init_latents": init_latents,
                "text_embeddings": text_embeddings
                }
        print(f"{filename}_{i}.png")
        latents = denoise([], 0, config, return_all_samples=True, generator=generator)
        latents = torch.cat(latents)
        torch.save(latents, os.path.join(dst_path , f"{filename}_{i}.pt"))