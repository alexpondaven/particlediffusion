# Data generation for experiments
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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
from src.embedding import CNN16, CNN64, Average, AverageDim, VAEAverage, Edges, init_weights, Style, VGG, RuleOfThirds, VGGRo3

import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import argparse

parser = argparse.ArgumentParser(description="Running diversity steps experiment.")
parser.add_argument("--mode", type=str, default="artist_max", help="type of dataset to generate")
parser.add_argument("--subject", type=str, default="tree", help="prompt name")
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
    "tree": "a beautiful painting of a tree with autumn flowers on a green hill by the river"
}

mode = args.mode
subject = args.subject

def gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed):
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
        
        artists = pd.read_csv("data/styles/artists.csv",header=None)
        for a, artist in enumerate(artists[0]):
            if a==artist_num:
                dst_path = os.path.join("data","final", subject,mode, artist.replace(" ", "_"))
                os.makedirs(dst_path, exist_ok=True)
                prompt = f"{prompt_subjects[subject]}, by {artist}"
                init_latents, text_embeddings = get_score_input(prompt, config, generator=generator, device=device, dtype=dtype)
                config = {**config,
                        "init_latents": init_latents,
                        "text_embeddings": text_embeddings
                        }
                steps = Steps(init_method="repulsive_no_noise") #repulsive_no_noise
                latents = denoise_particles(
                    config, generator, num_particles=numparticles, steps=steps.steps,
                    correction_step_type="auto",
                    addpart_level=addpart_level,
                    model=model, 
                    repulsive_strength=repulsive_strength, repulsive_strat="kernel"
                )
                # Save results
                imgs = decode_latent(latents, pipe.vae)
                imgs = output_to_img(imgs)
                imgs = (imgs * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in imgs]
                for i, pil_image in enumerate(pil_images):
                    filename = f"artist{a}_{seg*numparticles + i}"
                    pil_image.save(os.path.join(dst_path , f"{filename}.png"))


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
    artist_num=0
    numparticles=1000
    for seg in range(10):
        seed=1000+seg
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
            if a==artist_num:
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
                    filename = f"artist{a}_{seg*numparticles + i}"
                    pil_image.save(os.path.join(dst_path , f"{filename}.png"))
elif mode=="averagedim_all_r10000":
    ############### AVERAGEDIM ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_conv_act=True)
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    artist_num=0
    numparticles=1000
    segments=10
    repulsive_strength = 10000
    single_initial_latent=False
    init_seed=1000
    gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed)
    
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
    artist_num=0
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=2000
    gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed)
    
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
    artist_num=0
    numparticles=500
    segments=20
    repulsive_strength = 10000
    single_initial_latent=False
    init_seed=4000
    gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed)
elif mode=="ro3_all_r500":
    ############### Rule of thirds ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    model = RuleOfThirds().to(torch.device("cuda"))
    artist_num=0
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=3000
    gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed)
elif mode=="cnn16_all_r500":
    ############### CNN16 ALL RANDOM INITIAL LATENTS
    # Can only repulse each subset of 1k particles
    # Change seed just in case
    # One artist 10k
    model = CNN16(relu=True)
    model_path = "model16.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
    artist_num=0
    numparticles=500
    segments=20
    repulsive_strength = 500
    single_initial_latent=False
    init_seed=5000
    gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed)
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
    artist_num=0
    numparticles=500
    segments=20
    repulsive_strength = 1000
    single_initial_latent=False
    init_seed=6000
    gen_repulse(model, artist_num, numparticles, segments, repulsive_strength, single_initial_latent,init_seed)

