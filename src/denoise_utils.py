import torch
from tqdm.auto import tqdm
import numpy as np
from collections import deque, defaultdict
from functools import partial

from src.sampling_utils import random_step, langevin_step, repulsive_step_parallel, repulsive_step_series
from src.kernel import RBF
from src.score_utils import get_score, step_score, get_score_multiprompt

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
        score_list = []
        
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
        if return_all_samples:
            latent_list.append(latents)
            score_list.append(score)
            
    
    if return_all_samples:
        return latent_list, score_list
    
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
    noise_cond=None,
    weights=[],
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
                                                step_size=step_size, repulsive_strength=repulsive_strength,repulsive_strat=repulsive_strat,add_noise=add_noise,
                                                noise_cond=noise_cond)
            # with torch.no_grad():
            #     print(f"Correction {i} spread: SD - {spread(particles)} embedding - {spread(model(particles))}")
    elif correction_method=="repulsive_series" or correction_method=="repulsive_series_no_noise":
        add_noise = (correction_method=="repulsive_series")
        # Parallel particles
        phi_history = None
        for i in range(correction_steps):
            particles = repulsive_step_series(particles, phi_history, model, kernel, generator, sigma, t, config,
                                                step_size=step_size, repulsive_strength=repulsive_strength,repulsive_strat=repulsive_strat,add_noise=add_noise)
            # with torch.no_grad():
            #     print(f"Correction {i} spread: SD - {spread(particles)} embedding - {spread(model(particles))}")
    # elif correction_method=="repulsive_series":
    #     # TODO: Update particles one at a time
    #     # Keep track of (latent, score, pushforward gradient)
    #     phi_history = defaultdict(partial(deque, maxlen=phi_history_size))
    #     # Add particles 
    #     for particle in particles:
    #         phi_grad = get_phi_grad(particle, model)
    #         phi_history.append(phi_grad)

    #     for i in range(correction_steps):
    #         for n in range(len(particles)):
    #             score = get_score(particles[n], sigma, t, config)
    #             particles[n] = repulsive_step(particles[n], scores, phi_grad, phi_history, model, kernel, generator, step_size=step_size)
    #             print(f"Correction {i} spread: SD - {spread(particles)} embedding - {spread([model(particle) for particle in particles])}")
    #     new_particles = particles

    elif correction_method in ["langevin", "langevin_mask", "langevin_multiprompt", "score", "score_mask", "score_multiprompt"]:
        add_noise = "langevin" in correction_method
        masked = "mask" in correction_method
        multi_prompt = "multiprompt" in correction_method
        for _ in range(correction_steps):
            if multi_prompt:
                score = get_score_multiprompt(particles, sigma, t, config, weights)
            else:
                score = get_score(particles, sigma, t, config)
            particles = langevin_step(particles, score, generator, step_size=step_size, add_noise=add_noise, 
                                      masked = masked, noise_mask = config['noise_mask'])
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
    addpart_method="random",
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
        noise_cond = torch.ones(len(particles), device=device) / len(config['timesteps'])

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
                repulsive_strat=repulsive_strat,
                noise_cond=noise_cond,
                weights=steps.steps[i][2][0] # hacky to extract weights of ith step
            )
        
        # Automatic step_size using sigmas
        if correction_step_type=="auto":
                # correction_step_size =  sigma**2 / sigmas[0]**2
                correction_step_size = sigma * (sigma - sigmas[step_index + 1]) # / correction_steps[idx]
                print(correction_step_size.item())

        # Steps
        if i in steps:
            for step_method, num_steps, *weights in steps[i]:
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
                    repulsive_strat=repulsive_strat,
                    noise_cond=noise_cond,
                    weights=weights[0],
                )
    
    return particles

def denoise_series(
    config,
    generator,
    steps,
    correction_step_size=0.1,
    correction_step_type="manual", # or auto
    addpart_step_size=0.2,
    addpart_method="random",
    history_size = 100,
    num_particles=1,
    model=None,
    kernel=RBF(),
    repulsive_strength=10,
    repulsive_strat="kernel",
    device="cuda",
):
    """ General function to take steps and add particles at different noise levels of diffusion IN SERIES
        steps (Dict[List]): queries are noise levels, values are list of tuples (step_method, num_steps)
        correction_step_type (str): "manual" (choose correction_step_size) or "auto" (decided by formula)
        config (Dict): info for diffusion
        generator: RNG generator - reset before calling this method
        num_particles=2: number of particles to add at addpart_level
        Model: Embedding model for repulsion
        model_path: path to embedding model weights
        Kernel=RBF: Kernel class to use for repulsive steps
        device="cuda",
    """
    # Checks
    if addpart_method=="repulsive":
        print("Cannot use repulsive steps to add new particles.")
        return None
    # Create num_particles initial seeds
    init_particles = config['init_latents'] * sigmas.max() # 1x4x64x64 or Nx4x64x64
    # If only one initial latent given, perturb data by small 
    if particles.shape[0]==1:
        new_particles = random_step(init_particles, num_steps-1, generator, step_size=addpart_step_size, device=device)
        init_particles = torch.cat([init_particles,new_particles])

    # standard deviation of the initial noise distribution
    sigmas = config['sigmas']

    # Denoise particles
    particles = []
    history = defaultdict(partial(deque, maxlen=history_size))
    for init_particle in init_particles:
        for i, t in enumerate(tqdm(config['timesteps'])):
            t = t.to(device)
            step_index = (config['timesteps'] == t).nonzero().item()
            sigma = sigmas[step_index]
            
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
        
    return torch.stack(particles)