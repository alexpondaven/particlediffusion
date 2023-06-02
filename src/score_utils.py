import torch
from tqdm.auto import tqdm
import numpy as np


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
    n = len(samples)
    batch_size = min(config['batch_size'],n)
    # if len(samples) % batch_size != 0:
    #     print("ERROR: Number of particles must be divisible by batch size")
    #     return

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

