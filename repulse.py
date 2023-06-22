# Generate samples taking langevin/random/repulsive steps from an initial latent at different noise levels
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import yaml
import numpy as np
import random
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import argparse
from PIL import Image
from datetime import datetime

from src.visualise import image_grid, latent_to_img, decode_latent, output_to_img
from src.kernel import RBF
from src.embedding import CNN16, CNN64, Average, AverageDim, VAEAverage, Edges, init_weights, Style, VGG, RuleOfThirds, VGGRo3, Latent, VGG_noise
from src.score_utils import get_sigmas, get_score_input
from src.denoise_utils import denoise_particles
from src.steps import Steps

# Arguments
parser = argparse.ArgumentParser(description="Running diversity steps experiment.")
# Initial latent info
parser.add_argument("--prompt", type=str, default="", help="prompt")
parser.add_argument("--seed", type=int, default=1024, help="random seed")

# Diversification method
parser.add_argument("--method", type=str, default="repulsive", help="random, langevin, or repulsive")
parser.add_argument("--strength", type=int, default=0, help="repulsive strength")

# Particle/repulsive method arguments
parser.add_argument("--nparticles", type=int, default=5, help="no. of particles")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--model", type=str, default="averagedim", 
                    help="embedding model. choose from: cnn16 (random cnn), averagedim (latent channel average), edges (half plane difference), ro3 (rule of thirds), vgg_noise (style classifier), vgg_noisero3 (style classifier with rule of thirds), latent (repulsion on pure latent space)")
parser.add_argument("--style", action='store_true', help="whether to get style of model")

# Evaluation
parser.add_argument("--report", action='store_true', help="whether to plot report images")

args = parser.parse_args()
prompt = args.prompt
seed = args.seed
method = args.method
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

##### PARAMS #############################################################
if prompt=="":
    # prompt = "child flying a kite at the beach, by Thomas Kinkade, Vincent Van Gogh, Leonid Afremov, Claude Monet, Edward Hopper, Norman Rockwell, William-Adolphe Bouguereau, Albert Bierstadt, John Singer Sargent, Pierre-Auguste Renoir, Frida Kahlo, John William Waterhouse, Winslow Homer, Walt Disney, Thomas Moran, Phil Koch, Paul Cézanne, Camille Pissarro, Erin Hanson, Thomas Cole, Raphael, Steve Henderson, Pablo Picasso, Caspar David Friedrich, Ansel Adams, Diego Rivera, Steve McCurry, Bob Ross, John Atkinson Grimshaw, Rob Gonsalves, Paul Gauguin, James Tissot, Edouard Manet, Alphonse Mucha, Alfred Sisley, Fabian Perez, Gustave Courbet, Zaha Hadid, Jean-Léon Gérôme, Carl Larsson, Mary Cassatt, Sandro Botticelli, Daniel Ridgway Knight, Joaquín Sorolla, Andy Warhol, Kehinde Wiley, Alfred Eisenstaedt, Gustav Klimt, Dante Gabriel Rossetti, Tom Thomson"
    # prompt = 'painting of a beautiful vase of flowers'
    # prompt = "beautiful tree"
    prompt = "an ultra detailed beautiful painting psychedelic professional portrait of a Dune 2021 character, centered, intricate, by conrad roset, greg rutkowski and makoto shinkai, trending on artstation, Realistic, Cinematic"
    # prompt = "fantasy map of a continent with diverse terrain, ultra-detailed, by Wilson McLean!, HD, D&D, 4k, 8k, high detail!, intricate, encyclopedia illustration"
    # prompt = 'a portrait of a parkour runner with a black tank top and white running pants, city setting, by Thomas Kinkade, Vincent Van Gogh, Leonid Afremov, Claude Monet, Edward Hopper, Norman Rockwell, William-Adolphe Bouguereau, Albert Bierstadt, John Singer Sargent, Pierre-Auguste Renoir, Frida Kahlo, John William Waterhouse, Winslow Homer, Walt Disney, Thomas Moran, Phil Koch, Paul Cézanne, Camille Pissarro, Erin Hanson, Thomas Cole, Raphael, Steve Henderson, Pablo Picasso, Caspar David Friedrich, Ansel Adams, Diego Rivera, Steve McCurry, Bob Ross, John Atkinson Grimshaw, Rob Gonsalves, Paul Gauguin, James Tissot, Edouard Manet, Alphonse Mucha, Alfred Sisley, Fabian Perez, Gustave Courbet, Zaha Hadid, Jean-Léon Gérôme, Carl Larsson, Mary Cassatt, Sandro Botticelli, Daniel Ridgway Knight, Joaquín Sorolla, Andy Warhol, Kehinde Wiley, Alfred Eisenstaedt, Gustav Klimt, Dante Gabriel Rossetti, Tom Thomson'
numparticles = args.nparticles
single_initial_latent = False
repulsive_strength = args.strength

###########################################################################

if single_initial_latent:
    num_init_latents = 1
    addpart_level = 0 # Add particles in lvl 0
else:
    num_init_latents = numparticles
    addpart_level = None # Don't add any more particles
batch_size = min(50, numparticles)

config = {
    "pipe": pipe,
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "num_train_timesteps": 1000,
    "num_init_latents": num_init_latents, # 1 or numparticles
    "batch_size": batch_size,
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
    model = CNN64(relu=True)
    model_path = "model.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
elif args.model=="cnn16":
    model = CNN16(relu=True)
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
elif args.model=="ro3":
    model=RuleOfThirds()
elif args.model=="vgg_noise":
    num_outputs = 20
    model = VGG_noise(num_outputs=num_outputs, logsoftmax=False, return_conv_act=True)
    model_path ='data/model_chk/artist_noise_classifier_epoch2000_final.pt'
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
elif "vgg_noise" in args.model:
    num_outputs = 20
    model = VGG_noise(num_outputs=num_outputs, logsoftmax=False, return_pre_fconv=True)
    model_path ='data/model_chk/artist_noise_classifier_epoch2000_final.pt'
    model.load_state_dict(torch.load(model_path))
    model = VGGRo3(vgg=model, mode=args.model[9:], noise_cond=True)
    model.to(torch.device("cuda"))
elif args.model=="vgg":
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_conv_act=True)
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))
elif "vgg" in args.model:
    num_outputs = 20
    model = VGG(num_outputs=num_outputs, logsoftmax=False, return_pre_fconv=True)
    model_path ='data/model_chk/artist_classifier_epoch100.pt'
    model.load_state_dict(torch.load(model_path))
    model = VGGRo3(vgg=model, mode=args.model[3:])
    model.to(torch.device("cuda"))
elif args.model=="latent":
    model = Latent()

if args.style:
    model = Style(model)

############## Denoise ##########################################
seed=1024
generator = torch.Generator("cuda").manual_seed(seed)

# Repulsion on prediction steps
steps = Steps(init_method="repulsive_no_noise")

# Use steps API to add steps at specific noise levels
# steps.add(5,"langevin",1)
# steps.add_list(list(range(1,10)),"langevin",[1]*9)
# steps.add_all(method,5)
# steps.add_list(list(range(10)),method,[2]*10)
# steps.add_list([0,1,2,3],method,[10,10,10,10])
# steps.add_list([5],method,[2])
particles = denoise_particles(
    config, generator, num_particles=numparticles, steps=steps.steps,
    correction_step_type="auto",
    addpart_level=addpart_level,
    model=model, 
    repulsive_strength=repulsive_strength, repulsive_strat="kernel"
)
model.return_conv_act=False
if type(model)==VGG_noise:
    lvl_input = torch.ones(len(particles), device=device) / len(config['timesteps'])
    print("Classifier prediction:", nn.Softmax(dim=1)(model(particles, lvl_input)).argmax(dim=1))
else:
    print("Classifier prediction:", nn.Softmax(dim=1)(model(particles)).argmax(dim=1))
######################################################################


# Decode latents
images = output_to_img(decode_latent(particles, pipe.vae))
images = (images * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]

save_grid=True
report=args.report

if save_grid:
    if report:
        # Display images in grid with max_cols columns
        max_cols = 9
        ncols = max_cols if numparticles > max_cols else numparticles
        nrows = int(np.ceil(numparticles / ncols))
        grid = image_grid(pil_images,nrows,ncols)

        # Save image grid
        method = args.model
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        foldername = f"data/report_results/portrait/{method}"
        os.makedirs(foldername, exist_ok=True)
        grid.save(f"{foldername}/{method}_r{repulsive_strength}.png")
    else:
        # Display images in grid with max_cols columns
        max_cols = 9
        ncols = max_cols if numparticles > max_cols else numparticles
        nrows = int(np.ceil(numparticles / ncols))
        grid = image_grid(pil_images,nrows,ncols)

        # Save image grid
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if method=="repulsive":
            method = "repulsive_"+args.model
        filename = f"data/outputs/out_{method}_{timestamp}.png"
        grid.save(filename)
else:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    if method=="repulsive":
        method = "repulsive_"+args.model
    filename = f"data/outputs/out_{method}_{timestamp}"
    os.makedirs(filename, exist_ok=True)
    for i, pil_image in enumerate(pil_images):
        pil_image.save(f"{filename}/out_{method}_{timestamp}_{i}.png")
