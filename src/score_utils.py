import torch
from tqdm.auto import tqdm
import numpy as np
from collections import deque

from src.sampling_utils import random_step, langevin_step, repulsive_step_parallel
from src.kernel import RBF
from src.embedding import CNN64, CNN16, init_weights
import matplotlib.pyplot as plt


## Initialisation
def get_sigmas(config, device="cuda"):
    beta_end=config['beta_end']
    beta_start=config['beta_start']
    num_train_timesteps=config['num_train_timesteps']
    num_inference_steps=config['num_inference_steps']

    # linear
    # betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    #scaled linear
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # set_timesteps - set sigmas for num_inference_steps noise levels by interpolating training sigmas
    timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

    timesteps = torch.from_numpy(timesteps).to(device)
    sigmas = torch.from_numpy(sigmas).to(device=device)
    
    return sigmas, timesteps

def get_score_input(prompt, config, generator, device="cuda", dtype=torch.float32, init_strat="normal"):
    """Return text embedding and initial latent i.e. the input to the diffusion model"""
    pipe = config['pipe']
    batch_size = config['num_init_latents']
    height = config['height']
    width = config['width']

    if type(prompt)!=list:
        prompt = [prompt]*batch_size

    text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    # max_length = text_input.input_ids.shape[-1]
    uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Init latents
    if init_strat=="normal":
        init_latents = torch.randn(
            (batch_size, pipe.unet.in_channels, height//8, width//8),
            generator=generator,
            device=device,
        ).to(dtype)
    elif init_strat=="uniform":
        lb, ub = -5,5
        init_latents = torch.rand(
            (batch_size, pipe.unet.in_channels, height//8, width//8),
            generator=generator,
            device=device,
        ).to(dtype)
        init_latents = (ub-lb) * init_latents + lb
    else:
        print("ERROR: INVALID LATENT INITIALISATION STRATEGY")
    return init_latents, text_embeddings


## Score/denoising helpers
def scale_input(sample, sigma):
    """Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm."""
    sample = sample / ((sigma**2 + 1) ** 0.5)
    return sample

def get_score(samples, sigma, t, config):
    """
    Get current ∇_x p(x|sigma)
        sample: x (Nx4x64x64)
        sigma: noise level
        t: timestep for that noise level
        config
    """
    batch_size = config['batch_size']
    n = len(samples)
    if len(samples) % batch_size != 0:
        print("ERROR: Number of particles must be divisible by batch size")
        return

    scores = []
    for i in range(0, n, batch_size):
        sample = samples[i:i+batch_size]
        # expand latents for cfg to avoid 2 forward passes
        latent_model_input = torch.cat([sample] * 2)
        latent_model_input = scale_input(latent_model_input, sigma)

        if config["num_init_latents"]==1:
            # One initial latent
            text_embeddings = torch.stack([config['text_embeddings'][0]] * batch_size + [config['text_embeddings'][1]] * batch_size)
        elif config["num_init_latents"]==n:
            # All latents initialised already
            text_embeddings = torch.cat([config['text_embeddings'][i:i+batch_size]] + [config['text_embeddings'][n+i:n+i+batch_size]])
        else:
            print("ERROR: Not implemented num_init_latents != 1 or numparticles")
            return

        # predict noise residual
        with torch.no_grad():
            noise_pred = config['pipe'].unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config['cfg'] * (noise_pred_text - noise_pred_uncond)

        # D (pred_original_sample)
        D = sample - sigma * noise_pred

        # score
        score = (D - sample) / (sigma**2)
        scores.append(score)

    return torch.cat(scores)
      
def step_score(
        sample: torch.FloatTensor,
        score: torch.FloatTensor,
        sigmas,
        sigma_index,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise)."""
        sigma = sigmas[sigma_index]
        derivative = - sigma * score

        dt = sigmas[sigma_index + 1] - sigma

        prev_sample = sample + derivative * dt

        return prev_sample

def denoise(
    correction_levels,
    correction_steps,
    config,
    generator,
    return_all_samples=False,
    device="cuda",
    ):
    """ Denoising function
        init_latents: Starting latent sample to denoise from
        correction_levels: list of noise levels where langevin steps should be done (towards MAP)
        correction_steps: how many steps to do in each noise level correction_levels
        sigmas: noise level for each timestep
        timestep: associated timesteps to pass to unet
        return_final: return final latent sample, otherwise return list of all samples (save every other latent)
    """
    # standard deviation of the initial noise distribution
    sigmas = config['sigmas']
    latents = config['init_latents'] * sigmas.max()

    if return_all_samples:
        latent_list = [latents]
    for i, t in enumerate(tqdm(config['timesteps'])):
        t = t.to(device)
        step_index = (config['timesteps'] == t).nonzero().item()
        sigma = sigmas[step_index]

        # Langevin steps
        if i in correction_levels:
            for _ in range(correction_steps):
                score = get_score(latents, sigma, t, config)
                latents = langevin_step(latents, score, generator)
        
        # Move to next marginal in diffusion
        score = get_score(latents, sigma, t, config)
        latents = step_score(latents, score, sigmas, step_index)
        if return_all_samples and i%2==0:
            latent_list.append(latents)

    
    if return_all_samples:
        return latent_list
    
    return latents

## Particle utils
def spread(particles):
    # Get L2 distances between particles
    spread=0
    n = len(particles)
    for i in range(n):
        for j in range(i+1,n):
            spread += torch.mean((particles[i]-particles[j])**2)
    
    return spread

# Particle diffusion
def correct_particles(
    particles,
    sigma,
    t,
    correction_steps,
    correction_method,
    config,
    generator,
    step_size=0.2,
    model=None,
    kernel=None,
    repulsive_strength=10,
    repulsive_strat="kernel",
    ):
    """ At certain noise scale (step t), apply correction steps to all particles
        particles: N particles in the diffusion process
        method (str): type of correction step "random", "langevin", "score", or "repulsive"
        model: Embedding model (must be defined if repulsive correct_type specified)
        K: RBF Kernel (must be defined if repulsive correct_type specified)
        TODO: Make sampler factory class
    """
    if correction_method=="random":
        particles = random_step(particles, correction_steps, generator, step_size=step_size)
    elif correction_method=="repulsive" or correction_method=="repulsive_no_noise":
        add_noise = (correction_method=="repulsive")
        # Parallel particles
        phi_history = None
        for i in range(correction_steps):
            scores = get_score(particles, sigma, t, config)
            particles = repulsive_step_parallel(particles, scores, phi_history, model, kernel, generator, 
                                                step_size=step_size, repulsive_strength=repulsive_strength,repulsive_strat=repulsive_strat,add_noise=add_noise)
            with torch.no_grad():
                print(f"Correction {i} spread: SD - {spread(particles)} embedding - {spread(model(particles))}")

    # elif correction_method=="repulsive_series":
        # TODO: Update particles one at a time
        # phi_history = deque([], phi_history_size)
        # # Add particles 
        # for particle in particles:
        #     phi_grad = get_phi_grad(particle, model)
        #     phi_history.append(phi_grad)

        # for i in range(correction_steps):
        #     for n in range(len(particles)):
        #         score = get_score(particles[n], sigma, t, config)
        #         particles[n] = repulsive_step(particles[n], scores, phi_grad, phi_history, model, kernel, generator, step_size=step_size)
        #         print(f"Correction {i} spread: SD - {spread(particles)} embedding - {spread([model(particle) for particle in particles])}")
        # new_particles = particles

    elif correction_method=="langevin" or correction_method=="score":
        add_noise = (correction_method=="langevin")
        for _ in range(correction_steps):
            score = get_score(particles, sigma, t, config)
            particles = langevin_step(particles, score, generator, step_size=step_size, add_noise=add_noise)
    else:
        print(f"ERROR: Correction step type: '{correction_method}' not implemented yet")
            
    return particles

def denoise_particles(
    config,
    generator,
    steps,
    correction_step_size=0.1,
    correction_step_type="manual", # or auto
    addpart_level=0,
    addpart_steps=1,
    addpart_step_size=0.2,
    addpart_method="langevin",
    num_particles=1,
    model=None,
    kernel=RBF(),
    repulsive_strength=10,
    repulsive_strat="kernel",
    device="cuda",
):
    """ General function to take steps and add particles at different noise levels of diffusion
        steps (Dict[List]): queries are noise levels, values are list of tuples (step_method, num_steps)
        correction_step_type (str): "manual" (choose correction_step_size) or "auto" (decided by formula)
        addpart_level (int): noise level index to add particles in
        addpart_steps (int): number of steps taken between particles
        addpart_method (str): method of steps for adding particles e.g. random, langevin, score (NOT repulsive)
        config (Dict): info for diffusion
        generator: RNG generator - reset before calling this method
        num_particles=2: number of particles to add at addpart_level
        Model: Embedding model for repulsion
        model_path: path to embedding model weights
        Kernel=RBF: Kernel class to use for repulsive steps
        device="cuda",
    """
    if addpart_method=="repulsive":
        print("Cannot use repulsive steps to add new particles.")
        return None
    # standard deviation of the initial noise distribution
    sigmas = config['sigmas']

    # Denoise particles
    particles = config['init_latents'] * sigmas.max() # 1x4x64x64
    for i, t in enumerate(tqdm(config['timesteps'])):
        t = t.to(device)
        step_index = (config['timesteps'] == t).nonzero().item()
        sigma = sigmas[step_index]

        # Create particles
        if i==addpart_level:
            # TODO: Optimise having to re-calculate gradient for duplicate particles each time
            particles = torch.concat([particles] + [particles[0].unsqueeze(0)] * (num_particles-1))
            particles = correct_particles(
                particles, 
                sigma, 
                t, 
                correction_steps=addpart_steps, 
                correction_method=addpart_method,
                config=config, 
                generator=generator, 
                step_size=addpart_step_size, 
                model=model, 
                kernel=kernel,
                repulsive_strength=repulsive_strength,
                repulsive_strat=repulsive_strat
            )
        
        # Automatic step_size using sigmas
        if correction_step_type=="auto":
                # correction_step_size =  sigma**2 / sigmas[0]**2
                correction_step_size = sigma * (sigma - sigmas[step_index + 1]) # / correction_steps[idx]
                print(correction_step_size.item())

        # Steps
        if i in steps:
            for step_method, num_steps in steps[i]:
                print(i)
                particles = correct_particles(
                    particles, 
                    sigma, 
                    t, 
                    num_steps, 
                    correction_method=step_method, 
                    config=config, 
                    generator=generator, 
                    step_size=correction_step_size, 
                    model=model, 
                    kernel=kernel,
                    repulsive_strength=repulsive_strength,
                    repulsive_strat=repulsive_strat
                )
    
    return particles