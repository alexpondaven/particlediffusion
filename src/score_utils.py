import torch
from tqdm.auto import tqdm
import numpy as np

from src.sampling_utils import random_step, langevin_step, repulsive_step
from src.kernel import RBF
from src.embedding import CNN, init_weights

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

def get_score_input(prompt, config, generator, device="cuda"):
    """Return text embedding and initial latent i.e. the input to the diffusion model"""
    pipe = config['pipe']
    batch_size = config['batch_size']
    height = config['height']
    width = config['width']

    text_input = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    max_length = text_input.input_ids.shape[-1]
    uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Init latents
    init_latents = torch.randn(
        (batch_size, pipe.unet.in_channels, height//8, width//8),
        generator=generator,
        device=device,
    )
    return init_latents, text_embeddings


## Score/denoising helpers
def scale_input(sample, sigma):
    """Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm."""
    sample = sample / ((sigma**2 + 1) ** 0.5)
    return sample

def get_score(sample, sigma, t, config):
    """
    Get current ∇_x p(x|sigma)
        sample: x
        sigma: noise level
        t: timestep for that noise level
        config
    """
    # expand latents for cfg to avoid 2 forward passes
    latent_model_input = torch.cat([sample] * 2)
    latent_model_input = scale_input(latent_model_input, sigma)

    # predict noise residual
    with torch.no_grad():
        noise_pred = config['pipe'].unet(latent_model_input, t, encoder_hidden_states=config['text_embeddings']).sample
    
    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + config['cfg'] * (noise_pred_text - noise_pred_uncond)

    # D (pred_original_sample)
    D = sample - sigma * noise_pred

    # score
    score = (D - sample) / (sigma**2)

    return score
      
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
    K=None,
    ):
    """ At certain noise scale (step t), apply correction steps to all particles
        particles: N particles in the diffusion process
        method (str): type of correction step "random", "langevin", "score", or "repulsive"
        model: Embedding model (must be defined if repulsive correct_type specified)
        K: RBF Kernel (must be defined if repulsive correct_type specified)
        TODO: Make sampler factory class
    """
    if correction_method=="random":
        new_particles = random_step(particles, correction_steps, generator, step_size=step_size)
    elif correction_method=="repulsive":
        for _ in range(correction_steps):
            scores = [get_score(particle, sigma, t, config) for particle in particles]
            particles = repulsive_step(particles, scores, model, K, generator, step_size=step_size)
        new_particles = particles
    elif correction_method=="langevin" or correction_method=="score":
        new_particles = []
        add_noise = (correction_method=="langevin")
        for particle in particles:
            for _ in range(correction_steps):
                score = get_score(particle, sigma, t, config)
                particle = langevin_step(particle, score, generator, step_size=step_size, add_noise=add_noise)
            new_particles.append(particle)
    else:
        print(f"ERROR: Correction step type: '{correction_method}' not implemented yet")
            
    return new_particles

def denoise_particles(
    config,
    generator,
    correction_levels=[],
    correction_steps=1,
    correction_step_size=0.2,
    correction_method="langevin",
    addpart_level=0,
    addpart_steps=1,
    addpart_step_size=0.2,
    addpart_method="langevin",
    num_particles=1,
    Model=CNN,
    model_path="model.pt",
    Kernel=RBF,
    device="cuda",
):
    """ General function to take steps and add particles at different noise levels of diffusion
        correction_levels (int or List[int]): noise level indices to do correction steps in
        correction_steps (int or List[int]): number of correction steps to take in each noise level
        correction_method (str or List[str]): method of correction e.g. random, langevin, score or repulsive 
        addpart_level (int): noise level index to add particles in
        addpart_steps (int): number of steps taken between particles
        addpart_method (str): method of steps for adding particles e.g. random, langevin, score or repulsive 
        config (Dict): info for diffusion
        generator: RNG generator - reset before calling this method
        num_particles=2: number of particles to add at addpart_level
        Model: Embedding model for repulsion
        model_path: path to embedding model weights
        Kernel=RBF: Kernel class to use for repulsive steps
        device="cuda",
    """
    # standard deviation of the initial noise distribution
    sigmas = config['sigmas']
    latents = config['init_latents'] * sigmas.max()

    # Embedding model for repulsive force
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device("cuda"))

    # Kernel
    K = Kernel()

    particles = [latents]
    for i, t in enumerate(tqdm(config['timesteps'])):
        t = t.to(device)
        step_index = (config['timesteps'] == t).nonzero().item()
        sigma = sigmas[step_index]

        # Create particles
        if i==addpart_level:
            for _ in range(num_particles-1):
                new_particle = correct_particles(
                    [particles[-1]], 
                    sigma, 
                    t, 
                    correction_steps=addpart_steps, 
                    correction_method=addpart_method, 
                    config=config, 
                    generator=generator, 
                    step_size=addpart_step_size, 
                    model=model, 
                    K=K
                )
                particles.append(new_particle[0])
        
        # Correction steps
        if i in correction_levels:
            particles = correct_particles(
                particles, 
                sigma, 
                t, 
                correction_steps, 
                correction_method=correction_method, 
                config=config, 
                generator=generator, 
                step_size=correction_step_size, 
                model=model, 
                K=K
            )
        
        # Move to next marginal in diffusion
        for n in range(len(particles)):
            score = get_score(particles[n], sigma, t, config)
            particles[n] = step_score(particles[n], score, sigmas, step_index)
    
    return particles