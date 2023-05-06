# Sampling step helpers
# TODO: Have a class for these steps and child classes - factory?

import torch

def langevin_step(
        sample: torch.FloatTensor,
        score,
        generator,
        step_size=0.2,
        add_noise=True,
        device="cuda",
    ):
        """Take noisy step towards score"""
        prev_sample = sample + step_size * score
        if add_noise:
            noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator).to(device)
            prev_sample += ((step_size * 2) ** 0.5) * noise
        return prev_sample
        

def random_step(
        samples,
        correction_steps,
        generator,
        step_size=0.2,
        device="cuda",
    ):
        """Take noisy step (same amount of noise as langevin step)
            samples (List[torch.FloatTensor]): List of particles to take ranodm steps
            correction_step: number of steps (or equivalently the amount of noise added)
        """
        prev_samples = []
        for sample in samples:
            noise = torch.randn(sample.shape, layout=sample.layout, device=device, generator=generator).to(device)
            prev_sample = sample + ((correction_steps * step_size * 2) ** 0.5) * noise
            prev_samples.append(prev_sample)
        return prev_samples

def repulsive_step(
        particles,
        scores,
        model,
        K,
        generator,
        add_noise = False,
        step_size=0.2,
        repulsive_scale=1000.0, # strength of repulsive term
        device="cuda",
    ):
        """ Take repulsive langevin steps for all particles 
            particles: list of latent samples
            score: list of scores for each latent sample
            model: embedding model for latents to smaller dim space
            K: Kernel function
        
        """
        phis = []
        grads = []
        for particle in particles:
            # Embed latent to smaller dimension
            phi = model(particle)
            phis.append(phi)

            # Compute gradient ∇_latent (phi) for pushforward gradient computation
            grad = torch.autograd.functional.jacobian(
                model, 
                particle.clone().detach().requires_grad_()
            )
            grads.append(grad)

        # Compute kernel gradients
        tphis = torch.stack(phis,axis=0)
        # Set bandwidth of RBF kernel with median heuristic
        K.bandwidth(tphis, tphis) 
        # Kernel grad shape: embedding_size x num_particles x num_particles
        kernel_grad = K.grad_first(tphis, tphis)
        kernel_grad_sum = torch.sum(kernel_grad, axis=-1)

        # Repulsive term
        n = len(particles)
        new_particles = []
        for i, phi in enumerate(phis):
            # Pushforward/Chain rule
            # TODO: Check this multiplication is right
            repulsive = torch.einsum('i,ijklm->jklm',kernel_grad_sum[:,i], grads[i]) / n

            # Langevin term
            noise = torch.randn(particles[i].shape, layout=particles[i].layout, device=device, generator=generator).to(device)

            # repulsive_norm = torch.norm(repulsive.reshape(repulsive.shape[0], -1), dim=-1).mean()
            # print(repulsive_norm.item())

            # Score + Repulsion
            new_particle = particles[i] + step_size * (scores[i] + repulsive_scale * repulsive)
            # ONLY Repulsion
            # new_particle = particles[i] + step_size * (repulsive_scale * repulsive)
            if add_noise:
                new_particle += ((step_size * 2) ** 0.5) * noise

            new_particles.append(new_particle)
        return new_particles