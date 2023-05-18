# Sampling step helpers
# TODO: Have a class for these steps and child classes - factory?

import torch

def langevin_step(
        samples,
        scores,
        generator,
        step_size=0.2,
        add_noise=True,
        device="cuda",
    ):
        """Take noisy step towards score"""
        new_samples = samples + step_size * scores
        if add_noise:
            noise = torch.randn(samples.shape, layout=samples.layout, device=device, generator=generator).to(device)
            new_samples += ((step_size * 2) ** 0.5) * noise
        return new_samples
        

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
        noise = torch.randn(samples.shape, layout=samples.layout, device=device, generator=generator).to(device)
        new_samples = samples + ((correction_steps * step_size * 2) ** 0.5) * noise
        return new_samples

def repulsive_step_parallel(
        particles,
        scores,
        phi_history,
        model,
        K,
        generator,
        repulse=True,
        add_noise = True,
        step_size=0.2,
        snr=1, # strength of repulsive term
        phi_history_size=50,
        device="cuda",
    ):
        """ Take repulsive langevin step. 
            particles: list of latent samples
            scores: list of scores for each latent sample
            phi_history: List of previous phi particles used
            model: embedding model for latents to smaller dim space
            K: Kernel function
            generator: generator
            repulse: whether to repulse with repulsion term, or condition (repulse=False) by going in the opposite direction
        
        """
        # Checks
        n = len(particles)
        if n<=1:
             print("WARNING: Cannot repulse 1 particle on its own, add more particles")
             return
        if n < phi_history_size:
            phi_history_size=n
        
        # Embed latent to smaller dimension
        with torch.no_grad():
            phi = model(particles)

        # Add to phi_history FIFO
        if phi_history:
            history_space = phi_history_size - phi_history.shape[0]
            if n > history_space:
                phi_history = torch.cat((phi_history[n:], phi))
            else:
                phi_history = torch.cat((phi_history, phi))
        else:
            phi_history = phi

        # Compute gradient ∇_latent (phi) for pushforward gradient computation (shape= (NxE) x (Nx4x64x64)) E is output size of model
        # TODO: Should grad be found with all particles or each particle separate... I only need the grad of a phi with its own particle, not every other one
        grads = torch.autograd.functional.jacobian(
            model, 
            particles #.clone().detach().requires_grad_(), 
        )
        
        # Set bandwidth of RBF kernel with median heuristic
        K.bandwidth(phi_history, phi_history)
        print(f"Sigma: {K.sigma.item()}")
        # Kernel grad shape: embedding_size x num_particles x num_particles
        kernel_grad = K.grad_first(phi_history, phi_history)
        kernel_grad_sum = torch.sum(kernel_grad, axis=-1)

        # Repulsive term TODO: vectorize this for loop
        new_particles = []
        for i in range(n):
            # Pushforward/Chain rule
            # TODO: Check this multiplication is right
            repulsive = torch.einsum('i,ijklm->jklm',kernel_grad_sum[:,i-n], grads[i,:,i,...].unsqueeze(1)) / n

            # (debug) Get repulsion norm
            # repulsive_norm = torch.norm(repulsive.reshape(repulsive.shape[0], -1), dim=-1).mean()
            # score_norm = torch.norm(scores[i].reshape(scores[i].shape[0], -1), dim=-1).mean()
            # repulsive_scale = snr * score_norm / repulsive_norm
            # print(repulsive_norm.item())

            # (debug) Save repulsion term
            # torch.save(repulsive, f'repulsive{i}.pt')

            # Score + Repulsion
            if repulse:
                new_particle = particles[i] + step_size * (scores[i] - 100*repulsive)
            else:
                new_particle = particles[i] + step_size * (scores[i] + 100*repulsive)
            # ONLY Repulsion
            # new_particle = particles[i] - step_size * (repulsive_scale * repulsive)
            if add_noise:
                noise = torch.randn(particles[i].shape, layout=particles[i].layout, device=device, generator=generator).to(device)
                new_particle += ((step_size * 2) ** 0.5) * noise

            new_particles.append(new_particle)
        return torch.cat(new_particles,axis=0)