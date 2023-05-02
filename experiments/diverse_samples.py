# Generate samples taking langevin/random/repulsive steps from an initial latent at different noise levels
import os
import yaml
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import pandas as pd
import argparse

from src.kernel import RBF
from src.embedding import CNN, init_weights
from src.score_utils import get_sigmas, get_score_input, scale_input, get_score, step_score, denoise

# Arguments
parser = argparse.ArgumentParser(description="Running diversity steps experiment.")
# Initial latent info
parser.add_argument("--prompt", type=str, default="", help="prompt")
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Diversification method
parser.add_argument("--method", type=str, default="langevin", help="random, langevin, or repulsive")
parser.add_argument("--noise_level", type=int, default=0, help="noise_level to take steps in")
parser.add_argument("--num_steps", type=int, default=10, help="no. of steps to take between samples")
parser.add_argument("--num_samples", type=int, default=10, help="no. of samples to take")
parser.add_argument("--step_size", type=float, default=0.2, help="fixed stepsize")

# Particle/repulsive method arguments
parser.add_argument("--nparticles", type=int, default=2, help="no. of particles")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--model", type=str, default="rcnn", help="embedding model for latents for latent evaluation")
parser.add_argument("--gpu", type=int, default=0, help="gpu")

args = parser.parse_args()
prompt = args.prompt
seed = args.seed
method = args.method
noise_level = args.noise_level
num_steps = args.num_steps
num_samples = args.num_samples
step_size = args.step_size
nparticles = args.nparticles
device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

prompt_filename = prompt.replace(" ", "_")
results_folder = f"../data/{method}/{prompt_filename}_{seed}/"
filename = f"{noise_level}_{num_steps}_{step_size}"

if args.kernel == "rbf":
    Kernel = RBF

if args.model =="rcnn":
    # Random CNN
    model_path = "model.pt"
    net = CNN()
    net.load_state_dict(torch.load(model_path))
    net.to(torch.device("cuda"))


# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None

config = {
    "pipe": pipe,
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "num_train_timesteps": 1000,
    "batch_size": 1,
    "cfg": 8,
    "beta_start": 0.00085,
    "beta_end": 0.012,
}

# Seeds
seed=1024
generator = torch.Generator("cuda").manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(0)
random.seed(0)

# Noise levels
sigmas, timesteps = get_sigmas(config, device=device)
init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device="cuda")
config = {**config,
          "sigmas": sigmas,
          "timesteps": timesteps,
          "init_latents": init_latents,
          "text_embeddings": text_embeddings
          }

# Denoise

# Store image