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
parser.add_argument("--noise_level", type=int, default=0, help="noise_level to take steps in")
parser.add_argument("--num_steps", type=int, default=100, help="no. of steps to take between samples")
parser.add_argument("--num_samples", type=int, default=10, help="no. of samples to take")
parser.add_argument("--step_size", type=float, default=0.1, help="fixed stepsize")

# Particle/repulsive method arguments
parser.add_argument("--nparticles", type=int, default=2, help="no. of particles")
parser.add_argument("--kernel", type=str, default="rbf", help="kernel")
parser.add_argument("--model", type=str, default="averagedim", help="embedding model for latents for latent evaluation")
parser.add_argument("--style", action='store_true', help="whether to get style of model")
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

##### PARAMS #############################################################
prompt = "a humongous cave opening in a rainforest, waterfall in the middle, concept art, by Thomas Kinkade, Vincent Van Gogh, Leonid Afremov, Claude Monet, Edward Hopper, Norman Rockwell, William-Adolphe Bouguereau, Albert Bierstadt, John Singer Sargent, Pierre-Auguste Renoir, Frida Kahlo, John William Waterhouse, Winslow Homer, Walt Disney, Thomas Moran, Phil Koch, Paul Cézanne, Camille Pissarro, Erin Hanson, Thomas Cole, Raphael, Steve Henderson, Pablo Picasso, Caspar David Friedrich, Ansel Adams, Diego Rivera, Steve McCurry, Bob Ross, John Atkinson Grimshaw, Rob Gonsalves, Paul Gauguin, James Tissot, Edouard Manet, Alphonse Mucha, Alfred Sisley, Fabian Perez, Gustave Courbet, Zaha Hadid, Jean-Léon Gérôme, Carl Larsson, Mary Cassatt, Sandro Botticelli, Daniel Ridgway Knight, Joaquín Sorolla, Andy Warhol, Kehinde Wiley, Alfred Eisenstaedt, Gustav Klimt, Dante Gabriel Rossetti, Tom Thomson"
numparticles = 10
single_initial_latent = False

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

steps = Steps(init_method="repulsive_no_noise") #repulsive_no_noise
# steps.add_all(method,2)
# steps.add_list(list(range(10,20)),method,[10]*10)
# steps.add_list([0,1,2,3],method,[10,10,10,10])
# steps.add_list([5],method,[2])
particles = denoise_particles(
    config, generator, num_particles=numparticles, steps=steps.steps,
    correction_step_type="auto",
    addpart_level=addpart_level,
    model=model, 
    repulsive_strength=1000, repulsive_strat="kernel"
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

save_grid=False

if save_grid:
    # Display images in grid with max_cols columns
    max_cols = 100
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
