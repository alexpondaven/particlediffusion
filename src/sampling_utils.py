# Sampling step helpers
# TODO: Have a class for these steps and child classes - factory?

import torch

def langevin_step(
        sample: torch.FloatTensor,
        score,
        generator,
        step_size=0.2,
        device="cuda",
    ):
        """Take noisy step towards score"""
        noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator).to(device)
        prev_sample = sample + step_size * score + ((step_size * 2) ** 0.5) * noise
        return prev_sample

def random_step(
        sample: torch.FloatTensor,
        score,
        generator,
        step_size=0.2,
        device="cuda",
    ):
        """Take noisy step (same amount of noise as langevin step)
        - Used to compare langevin step to random step
        """
        noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator).to(device)
        prev_sample = sample + ((step_size * 2) ** 0.5) * noise
        return prev_sample

def repulsive_step(
        sample: torch.FloatTensor,
        score,
        generator,
        step_size=0.2,
        device="cuda",
    ):
        """ TODO Take repulsive langevin step. """
        noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator).to(device)

        # Get kernel gradient and embedding gradient

        prev_sample = sample + step_size * score + ((step_size * 2) ** 0.5) * noise
        return prev_sample