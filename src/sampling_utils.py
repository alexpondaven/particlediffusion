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

def repulsive_step_parallel(
        particles,
        scores,
        phi_history,
        model,
        K,
        generator,
        add_noise = True,
        step_size=0.2,
        snr=1, # strength of repulsive term
        device="cuda",
    ):
        """ Take repulsive langevin step. 
            particles: list of latent samples
            scores: list of scores for each latent sample
            phi_history: List of previous phi particles used
            model: embedding model for latents to smaller dim space
            K: Kernel function
        
        """
        grads = []
        for particle in particles:
            # Embed latent to smaller dimension
            phi = model(particle)
            phi_history.append(phi)

            # Compute gradient ∇_latent (phi) for pushforward gradient computation
            grad = torch.autograd.functional.jacobian(
                model, 
                particle.clone().detach().requires_grad_()
            )
            grads.append(grad)

        # Compute kernel gradients
        tphis = torch.stack(list(phi_history),axis=0)
        # Set bandwidth of RBF kernel with median heuristic
        K.bandwidth(tphis, tphis)
        print(f"Sigma: {K.sigma.item()}")
        # Kernel grad shape: embedding_size x num_particles x num_particles
        kernel_grad = K.grad_first(tphis, tphis)
        kernel_grad_sum = torch.sum(kernel_grad, axis=-1)

        # Repulsive term
        nparticles = len(particles)
        new_particles = []
        for i in range(nparticles):
            # Pushforward/Chain rule
            # TODO: Check this multiplication is right
            repulsive = torch.einsum('i,ijklm->jklm',kernel_grad_sum[:,i-nparticles], grads[i]) / nparticles

            # (debug) Get repulsion norm
            # repulsive_norm = torch.norm(repulsive.reshape(repulsive.shape[0], -1), dim=-1).mean()
            # score_norm = torch.norm(scores[i].reshape(scores[i].shape[0], -1), dim=-1).mean()
            # repulsive_scale = snr * score_norm / repulsive_norm
            # print(repulsive_norm.item())

            # (debug) Save repulsion term
            # torch.save(repulsive, f'repulsive{i}.pt')

            # Score + Repulsion
            new_particle = particles[i] + step_size * (scores[i] - 1000*repulsive)
            # ONLY Repulsion
            # new_particle = particles[i] - step_size * (repulsive_scale * repulsive)
            if add_noise:
                noise = torch.randn(particles[i].shape, layout=particles[i].layout, device=device, generator=generator).to(device)
                new_particle += ((step_size * 2) ** 0.5) * noise

            new_particles.append(new_particle)
        return new_particles