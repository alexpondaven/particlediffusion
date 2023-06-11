# Data generation for experiments
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from src.score_utils import get_sigmas, get_score_input, scale_input, get_score, step_score
from src.sampling_utils import random_step, langevin_step, repulsive_step_parallel
from src.kernel import RBF
from src.embedding import CNN64, CNN16, init_weights, AverageDim, Average, VAEAverage, Style
from src.denoise_utils import denoise_particles, denoise
from src.steps import Steps
from src.embedding import CNN16, CNN64, Average, AverageDim, VAEAverage, Edges, init_weights, Style, VGG, RuleOfThirds, VGGRo3, VGG_noise

import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import argparse

parser = argparse.ArgumentParser(description="Running diversity steps experiment.")
parser.add_argument("--mode", type=str, default="min_div", help="type of dataset to generate")
parser.add_argument("--subject", type=str, default="cave", help="prompt name")
parser.add_argument("--artistnum", type=int, default=-2, help="number of artist (-1 if no artist, -2 if all artist)")
args = parser.parse_args()

# Using 512x512 resolution
model_id = "stabilityai/stable-diffusion-2-base"
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda"
dtype=torch.float32
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.safety_checker = None
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

prompt_subjects = {
    "cave": "a humongous cave opening in a rainforest, waterfall in the middle, concept art",
    "tree": "a beautiful painting of a tree with autumn flowers on a green hill by the river", #https://prompthero.com/prompt/9645f10f44e
    "vase": "painting of a beautiful vase of flowers",
    "kite": "child flying a kite at the beach", #https://openart.ai/discovery/sd-1007129292574040064
    "parkour": "a portrait of a parkour runner with a black tank top and white running pants, city setting",
    "actress": "A breathtaking full body portrait of Ana de Armas, clothed, reminiscent of classical Renaissance paintings, with soft, luminous lighting, and delicate brushwork. Timeless and evocative" # https://stable-diffusion-art.com/chatgpt-prompt/ + chatgpt

}

mode = args.mode
subject = args.subject
artistnum = args.artistnum

def gen_repulse(numparticles, segments, single_initial_latent,init_seed, repulsive=True, model=None, repulsive_strength=None, langevin=False):
    # For each segment, generate numparticles of artist_num using kernel repulsion in model space
    if single_initial_latent:
        num_init_latents = 1
        addpart_level = 0 # Add particles in lvl 0
    else:
        num_init_latents = numparticles
        addpart_level = None # Don't add any more particles
    
    # Denoise all subsets (repulsion within subsets)
    for seg in range(segments):
        seed=init_seed+seg
        generator = torch.Generator(device).manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Settings
        config = {
            "pipe": pipe,
            "height": 512,
            "width": 512,
            "num_inference_steps": 20,
            "num_train_timesteps": 1000,
            "num_init_latents": num_init_latents,
            "batch_size": 50,
            "cfg": 8,
            "beta_start": 0.00085,
            "beta_end": 0.012,
        }
        # Noise levels
        sigmas, timesteps = get_sigmas(config, device=device)
        config = {**config,
                "sigmas": sigmas,
                "timesteps": timesteps,
                }
        
        # Name and prompt
        if artistnum==-1:
            foldername = "base"
            filename = "base"
            prompt = f"{prompt_subjects[subject]}"
        elif artistnum==-2:
            foldername = "all"
            filename = "all"
            prompt = f"{prompt_subjects[subject]}, by Thomas Kinkade, Vincent Van Gogh, Leonid Afremov, Claude Monet, Edward Hopper, Norman Rockwell, William-Adolphe Bouguereau, Albert Bierstadt, John Singer Sargent, Pierre-Auguste Renoir, Frida Kahlo, John William Waterhouse, Winslow Homer, Walt Disney, Thomas Moran, Phil Koch, Paul Cézanne, Camille Pissarro, Erin Hanson, Thomas Cole, Raphael, Steve Henderson, Pablo Picasso, Caspar David Friedrich, Ansel Adams, Diego Rivera, Steve McCurry, Bob Ross, John Atkinson Grimshaw, Rob Gonsalves, Paul Gauguin, James Tissot, Edouard Manet, Alphonse Mucha, Alfred Sisley, Fabian Perez, Gustave Courbet, Zaha Hadid, Jean-Léon Gérôme, Carl Larsson, Mary Cassatt, Sandro Botticelli, Daniel Ridgway Knight, Joaquín Sorolla, Andy Warhol, Kehinde Wiley, Alfred Eisenstaedt, Gustav Klimt, Dante Gabriel Rossetti, Tom Thomson"
        else:
            artists = pd.read_csv("data/styles/artists.csv",header=None)
            artist = artists[0][artistnum]
            foldername = artist.replace(" ", "_")
            filename = f"artist{artistnum}"
            prompt = f"{prompt_subjects[subject]}, by {artist}"
        dst_path = os.path.join("data","final", subject,mode, foldername)
        os.makedirs(dst_path, exist_ok=True)

        # Denoise
        init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device=device, dtype=dtype)
        config = {**config,
                "init_latents": init_latents,
                "text_embeddings": text_embeddings
                }
        if repulsive:
            steps = Steps(init_method="repulsive_no_noise") #repulsive_no_noise
            if langevin:
                # Note: not experimenting with repulsive langevin steps
                steps.add_all("repulsive",1)
            latents = denoise_particles(
                config, generator, num_particles=numparticles, steps=steps.steps,
                correction_step_type="auto",
                addpart_level=addpart_level,
                model=model, 
                repulsive_strength=repulsive_strength, repulsive_strat="kernel"
            )
        else:
            steps = Steps(init_method="score")
            if langevin:
                steps.add_list(list(range(10)),"langevin",[1]*10)
                # steps.add_all("langevin",1)
            latents = denoise_particles(
                    config, generator, num_particles=numparticles, steps=steps.steps,
                    correction_step_type="auto",
                    addpart_level=addpart_level,
                )
        # Save results
        imgs = decode_latent(latents, pipe.vae)
        imgs = output_to_img(imgs)
        imgs = (imgs * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in imgs]
        for i, pil_image in enumerate(pil_images):
            pil_image.save(os.path.join(dst_path , f"{filename}_{seg*numparticles + i}.png"))


if mode=="max_div":
    ############# MAX DIVERSITY
    # Seeds
    seed=0
    generator = torch.Generator(device).manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Generate 50 artists, 200 seeds each
    numparticles=200
    # Settings
    config = {
        "pipe": pipe,
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "num_train_timesteps": 1000,
        "num_init_latents": numparticles,
        "batch_size": 50,
        "cfg": 8,
        "beta_start": 0.00085,
        "beta_end": 0.012,
    }
    steps = Steps(init_method="score")
    # Noise levels
    sigmas, timesteps = get_sigmas(config, device=device)
    config = {**config,
            "sigmas": sigmas,
            "timesteps": timesteps,
            }
    
    artists = pd.read_csv("data/styles/artists.csv",header=None)
    for a, artist in enumerate(artists[0]):
        dst_path = os.path.join("data","final", subject, mode, artist.replace(" ", "_"))
        os.makedirs(dst_path, exist_ok=True)
        prompt = f"{prompt_subjects[subject]}, by {artist}"
        init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device=device, dtype=dtype)
        config = {**config,
                "init_latents": init_latents,
                "text_embeddings": text_embeddings
                }
        latents = denoise_particles(
            config, generator, num_particles=numparticles, steps=steps.steps,
            correction_step_type="auto",
            addpart_level=None,
        )
        # Save results
        imgs = decode_latent(latents, pipe.vae)
        imgs = output_to_img(imgs)
        imgs = (imgs * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in imgs]
        for i, pil_image in enumerate(pil_images):
            filename = f"artist{a}_{i}"
            pil_image.save(os.path.join(dst_path , f"{filename}.png"))
elif mode=="max_div2":
    ############### Change seed
    seed=1
    generator = torch.Generator(device).manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Generate 50 artists, 200 seeds each
    numparticles=200
    # Settings
    config = {
        "pipe": pipe,
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "num_train_timesteps": 1000,
        "num_init_latents": numparticles,
        "batch_size": 50,
        "cfg": 8,
        "beta_start": 0.00085,
        "beta_end": 0.012,
    }
    steps = Steps(init_method="score")
    # Noise levels
    sigmas, timesteps = get_sigmas(config, device=device)
    config = {**config,
            "sigmas": sigmas,
            "timesteps": timesteps,
            }
    
    artists = pd.read_csv("data/styles/artists.csv",header=None)
    for a, artist in enumerate(artists[0]):
        dst_path = os.path.join("data","final", subject,mode, artist.replace(" ", "_"))
        os.makedirs(dst_path, exist_ok=True)
        prompt = f"{prompt_subjects[subject]}, by {artist}"
        init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device=device, dtype=dtype)
        config = {**config,
                "init_latents": init_latents,
                "text_embeddings": text_embeddings
                }
        latents = denoise_particles(
            config, generator, num_particles=numparticles, steps=steps.steps,
            correction_step_type="auto",
            addpart_level=None,
        )
        # Save results
        imgs = decode_latent(latents, pipe.vae)
        imgs = output_to_img(imgs)
        imgs = (imgs * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in imgs]
        for i, pil_image in enumerate(pil_images):
            filename = f"artist{a}_{i}"
            pil_image.save(os.path.join(dst_path , f"{filename}.png"))
elif mode=="min_div":
    ############### MINIMUM DIVERSITY
    # Change seed just in case
    # One artist 10k
    # Note: original base min_div for first 3 prompts had seed 9000
    numparticles=1000
    segments = 10
    single_initial_latent=False # TODO put this outside as argument to script
    init_seed = 1000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, repulsive=False)
elif mode=="langevin":
    ############### MINIMUM DIVERSITY WITH LANGEVIN
    # Change seed just in case
    # One artist 10k
    # Note: original base min_div for first 3 prompts had seed 9000
    numparticles=1000
    segments = 10
    single_initial_latent=False
    init_seed = 1000 # CHANGE BACK TO 8000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, repulsive=False, langevin=True)
elif mode=="averagedim_all_r1000":
    ############### AVERAGEDIM ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_conv_act=True)
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    numparticles=1000
    segments=10
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=1000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
    
elif mode=="vggro3_all_r1000":
    ############### AVERAGEDIM ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_pre_fconv=True) # Get spatial filters
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model = VGGRo3(vgg=model, mode="ro3")
    model.to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=2000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
    
elif mode=="vgg_all_r10000":
    ############### VGG ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_conv_act=True)
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 10000
    single_initial_latent=False
    init_seed=4000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
elif mode=="ro3_all_r1000":
    ############### Rule of thirds ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    model = RuleOfThirds().to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=3000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
elif mode=="cnn16_all_r1000":
    ############### CNN16 ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    model = CNN16(relu=True)
    model_path = "model16.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=5000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
elif mode=="vgg_dc_channel_all_r1000":
    ############### Rule of thirds ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_pre_fconv=True) # Get spatial filters
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model = VGGRo3(vgg=model, mode="dc_channel")
    model.to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=6000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
elif mode=="vgg_noise_all_r1000":
    ############### VGG NOISE
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG_noise(num_outputs=num_outputs, logsoftmax=False, return_conv_act=True)
    model_path ='data/model_chk/artist_noise_classifier_epoch2000_final.pt'
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=4000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)
elif mode=="vgg_noisero3_all_r1000":
    ############### VGG NOISE WITH RO3
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG_noise(num_outputs=num_outputs, logsoftmax=False, return_pre_fconv=True)
    model_path ='data/model_chk/artist_noise_classifier_epoch2000_final.pt'
    model.load_state_dict(torch.load(model_path))
    model = VGGRo3(vgg=model, mode="ro3", noise_cond=True)
    model.to(torch.device("cuda"))
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=4000
    gen_repulse(numparticles, segments, single_initial_latent,init_seed, model=model, repulsive_strength=repulsive_strength)

