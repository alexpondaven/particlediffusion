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


## Sampling helpers
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

def langevin_step(
        sample: torch.FloatTensor,
        score,
        generator,
        snr = 0.01,
        device="cuda",
    ):
        """Take noisy step towards score - function based off VE shceduler corrector step method"""
        # For small batch sizes, the paper "suggest replacing norm(z) with sqrt(d), where d is the dim. of z"
        # sample noise for correction
        noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator).to(device)
        # offset_noise = 0.1 * torch.randn((latents.shape[0], latents.shape[1], 1, 1), layout=sample.layout, device=device, generator=generator)
        # noise = (noise + offset_noise).to(device)

        # compute step size from the model_output, the noise, and the snr
        # grad_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
        # noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        # step_size = (snr * noise_norm / grad_norm) ** 2 * 2
        # step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # step_size = step_size.flatten()
        # while len(step_size.shape) < len(sample.shape):
        #     step_size = step_size.unsqueeze(-1)

        step_size=0.1
        
        prev_sample_mean = sample + step_size * score
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise

        return prev_sample

def repulsive_term(particles):
    # Repulsive term using random CNN diversity calculation
    pass

def denoise(
    score_move,
    config,
    return_all_samples=False,
    save_score_norms=False,
    device="cuda",
    ):
    """ Denoising function
        init_latents: Starting latent sample to denoise from
        score_move: list of noise levels where score updates should be done (towards MAP)
        sigmas: noise level for each timestep
        timestep: associated timesteps to pass to unet
        return_final: return final latent sample, otherwise return list of all samples
    """
    # standard deviation of the initial noise distribution
    sigmas = config['sigmas']
    latents = config['init_latents'] * sigmas.max()

    if return_all_samples:
        latent_list = [latents]
    score_norm_hist = []
    for i, t in enumerate(tqdm(config['timesteps'])):
        t = t.to(device)
        step_index = (config['timesteps'] == t).nonzero().item()
        sigma = sigmas[step_index]
        
        # Move to next marginal in diffusion
        score = get_score(latents, sigma, t, config)
        latents = step_score(latents, score, sigmas, step_index)
        if return_all_samples:
            latent_list.append(latents)
        
        # Langevin steps at certain noise level
        if i in score_move:
            for _ in range(10):
                score = get_score(latents, sigma, t, config)
                score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                if save_score_norms:
                    score_norm_hist.append(score_norm.item())
                # latents = langevin_step(latents, score, snr=0.05, generator=generator)
                latents += 0.1*score
                if return_all_samples:
                    latent_list.append(latents)

    
    if return_all_samples:
        return latent_list, score_norm_hist
    
    return latents, score_norm_hist