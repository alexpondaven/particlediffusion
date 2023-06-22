import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
from diffusers import DiffusionPipeline
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import glob
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

from fid.fid_score import calculate_activation_statistics, save_statistics, load_statistics, calculate_frechet_distance, get_activations
from fid.inception import InceptionV3

from PIL import Image
from tqdm.auto import tqdm
from torch import autocast

from src.visualise import image_grid, latent_to_img, decode_latent, output_to_img

# Setup
seed=1024
generator = torch.Generator("cuda").manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(0)
random.seed(0)

# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
# model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)#, torch_dtype=torch.float16)
device = "cuda"
pipe = pipe.to(device)
pipe.safety_checker = None
scheduler_config = pipe.scheduler.config

# Diffusion using pipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DDIMScheduler, PNDMScheduler, ScoreSdeVeScheduler, UniPCMultistepScheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler_config)

seed=1024
generator = torch.Generator("cuda").manual_seed(seed)
dst_path = 'data/final/actress/eulera/all/'
os.makedirs(dst_path, exist_ok=True)
for n in range(2000):
    num_cols = 5

    
    # prompt = ["full length shot, super hero pose, biomechanical suit, inflatable shapes, wearing epic bionic cyborg implants, masterpiece, intricate, biopunk futuristic wardrobe, highly detailed, artstation, concept art, cyberpunk, octane render"]
    prompt = ["A breathtaking full body portrait of Ana de Armas, clothed, reminiscent of classical Renaissance paintings, with soft, luminous lighting, and delicate brushwork. Timeless and evocative, by Thomas Kinkade, Vincent Van Gogh, Leonid Afremov, Claude Monet, Edward Hopper, Norman Rockwell, William-Adolphe Bouguereau, Albert Bierstadt, John Singer Sargent, Pierre-Auguste Renoir, Frida Kahlo, John William Waterhouse, Winslow Homer, Walt Disney, Thomas Moran, Phil Koch, Paul Cézanne, Camille Pissarro, Erin Hanson, Thomas Cole, Raphael, Steve Henderson, Pablo Picasso, Caspar David Friedrich, Ansel Adams, Diego Rivera, Steve McCurry, Bob Ross, John Atkinson Grimshaw, Rob Gonsalves, Paul Gauguin, James Tissot, Edouard Manet, Alphonse Mucha, Alfred Sisley, Fabian Perez, Gustave Courbet, Zaha Hadid, Jean-Léon Gérôme, Carl Larsson, Mary Cassatt, Sandro Botticelli, Daniel Ridgway Knight, Joaquín Sorolla, Andy Warhol, Kehinde Wiley, Alfred Eisenstaedt, Gustav Klimt, Dante Gabriel Rossetti, Tom Thomson"]

    prompt = prompt* num_cols

    all_images = []
    images = pipe(prompt, num_inference_steps=20, generator=generator, guidance_scale=8).images
    # all_images.extend(images)

    # grid = image_grid(all_images, rows=1, cols=num_cols)
    for i, img in enumerate(images):
        img.save(os.path.join(dst_path, f"eulera_{i+n*num_cols}.png"))   

