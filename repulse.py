# Generate samples taking langevin/random/repulsive steps from an initial latent at different noise levels
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import yaml
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import argparse
from PIL import Image

from src.visualise import image_grid, latent_to_img, decode_latent, output_to_img
from src.kernel import RBF
from src.embedding import CNN16, CNN64, Average, AverageDim, VAEAverage, Edges, init_weights
from src.score_utils import get_sigmas, get_score_input, denoise_particles

# Arguments
parser = argparse.ArgumentParser(description="Running diversity steps experiment.")
# Initial latent info
parser.add_argument("--prompt", type=str, default="", help="prompt")
parser.add_argument("--seed", type=int, default=1024, help="random seed")

# Diversification method
parser.add_argument("--method", type=str, default="langevin", help="random, langevin, or repulsive")
parser.add_argument("--noise_level", type=int, default=0, help="noise_level to take steps in")
parser.add_argument("--num_steps", type=int, default=100, help="no. of steps to take between samples")
parser.add_argument("--num_samples", type=int, default=10, help="no. of samples to take")
parser.add_argument("--step_size", type=float, default=0.1, help="fixed stepsize")

# Particle/repulsive method arguments
parser.add_argument("--nparticles", type=int, default=2, help="no. of particles")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--model", type=str, default="averagedim", help="embedding model for latents for latent evaluation")
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

prompt = "eiffel tower"
config = {
    "pipe": pipe,
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "num_train_timesteps": 1000,
    "num_init_latents": 1, # 1 or num_particles
    "batch_size": 50,
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
init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device="cuda", init_strat="normal")
config = {**config,
          "sigmas": sigmas,
          "timesteps": timesteps,
          "init_latents": init_latents,
          "text_embeddings": text_embeddings
          }

# Embedding model for repulsive force
if args.model=="cnn64":
    model = CNN64(relu=False)
    model_path = "model.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
elif args.model=="cnn16":
    model = CNN16(relu=False)
    model_path = "model16.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
elif args.model=="vaeaverage":
    model=VAEAverage(vae=pipe.vae)
elif args.model=="averagedim":
    model=AverageDim()
elif args.model=="average":
    model=Average()
elif args.model=="edges":
    model=Edges()

# Denoise
seed=1024
generator = torch.Generator("cuda").manual_seed(seed)
numparticles=100
particles = denoise_particles(
    config, generator, num_particles=numparticles, 
    correction_levels= np.repeat(list(range(20)),2), 
    correction_steps=[10,1]*20, 
    correction_method=['repulsive','score']*20, # TODO: Remember to set repulse=True
    correction_step_type="auto",
    # addpart_level=None,
    model=model
)

# Decode latents
images = output_to_img(decode_latent(particles, pipe.vae))
images = (images * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]

# Display images in grid with max_cols columns
max_cols = 10
ncols = max_cols if numparticles > max_cols else numparticles
nrows = int(np.ceil(numparticles / ncols))
grid = image_grid(pil_images,nrows,ncols)

# Save image grid
filename = f"data/out_{method}.png"
grid.save(filename)
