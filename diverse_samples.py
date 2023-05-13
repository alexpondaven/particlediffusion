# Generate samples taking langevin/random/repulsive steps from an initial latent at different noise levels
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import yaml
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import pandas as pd
import argparse
from PIL import Image

from src.visualise import image_grid, latent_to_img, decode_latent, output_to_img
from src.kernel import RBF
from src.embedding import CNN64, CNN16, Average, SoftBoundedAverage, VAEAverage, init_weights
from src.score_utils import get_sigmas, get_score_input, denoise_particles

# Arguments
parser = argparse.ArgumentParser(description="Running diversity steps experiment.")
# Initial latent info
parser.add_argument("--prompt", type=str, default="", help="prompt")
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Diversification method
parser.add_argument("--method", type=str, default="langevin", help="random, langevin, or repulsive")
parser.add_argument("--noise_level", type=int, default=0, help="noise_level to take steps in")
parser.add_argument("--num_steps", type=int, default=100, help="no. of steps to take between samples")
parser.add_argument("--num_samples", type=int, default=10, help="no. of samples to take")
parser.add_argument("--step_size", type=float, default=0.1, help="fixed stepsize")

# Particle/repulsive method arguments
parser.add_argument("--nparticles", type=int, default=2, help="no. of particles")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--model", type=str, default="rcnn", help="embedding model for latents for latent evaluation")
parser.add_argument("--gpu", type=int, default=3, help="gpu")

args = parser.parse_args()
prompt = args.prompt
seed = args.seed
method = args.method
noise_level = args.noise_level
num_steps = args.num_steps
num_samples = args.num_samples
step_size = args.step_size
nparticles = args.nparticles
device="cuda"
# device = torch.device(f"cuda:{args.gpu}" if args.gpu != -1 else "cpu")

# prompt_filename = prompt.replace(" ", "_")
# results_folder = f"../data/{method}/{prompt_filename}_{seed}/"
# filename = f"{noise_level}_{num_steps}_{step_size}"

if args.kernel == "rbf":
    Kernel = RBF


# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
device = "cuda"
pipe = pipe.to(device)
pipe.safety_checker = None

prompt = "a black cat"
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
generator = torch.Generator("cuda").manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
seed=1024
generator = torch.Generator("cuda").manual_seed(seed)
# Embedding model for repulsive force
# model = CNN16(relu=False)
# model_path = "model16.pt"
# model.load_state_dict(torch.load(model_path))
# model.to(torch.device("cuda"))
model=VAEAverage(vae=pipe.vae)
# model=Average()
# Denoise
particles = denoise_particles(config, generator, num_particles=2, correction_levels=[0], correction_steps=20, correction_step_size=0.2,
                                correction_method="repulsive", model=model)
pil_images = []
for l in particles:
    image = output_to_img(decode_latent(l, pipe.vae))
    images = (image * 255).round().astype("uint8")
    pil_images.append([Image.fromarray(image) for image in images][0])

grid = image_grid(pil_images,1,len(particles))

file_step_size = str(step_size).replace(".","_")
filename = "data/test.png"
grid.save(filename)
