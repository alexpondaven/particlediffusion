# Sampling step helpers
# TODO: Have a class for these steps and child classes - factory?

import torch
# from torch.func import jacrev
from src.embedding import init_weights

def langevin_step(
        samples,
        scores,
        generator,
        step_size=0.2,
        add_noise=True,
        device="cuda",
    ):
        """Take noisy step towards score"""
        # new_samples = []
        # for i in range(len(samples)):
        #     new_sample = samples[i] + step_size * scores[i]
        #     if add_noise:
        #         noise = torch.randn(samples[i].shape, layout=samples[i].layout, device=device, generator=generator).to(device)
        #         new_sample += ((step_size * 2) ** 0.5) * noise
        #     new_samples.append(new_sample)
        # return torch.stack(new_samples)
             
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
        repulsive_strength=10, # strength of repulsive term
        phi_history_size=50,
        repulsive_strat = "kernel",
        device="cuda",
        weight_reset = True
    ):
        """ Take repulsive langevin step. 
            particles: list of latent samples
            scores: list of scores for each latent sample
            phi_history: List of previous phi particles used
            model: embedding model for latents to smaller dim space
            K: Kernel function
            generator: generator
            repulse: whether to repulse with repulsion term, or condition (repulse=False) by going in the opposite direction
            repulsive_strat (str): strategy of computing repulsive term
                "kernel"    kernel gradient
                "score"     use other particles scores
        
        """
        # Re-init weights every time
        if weight_reset:
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


        n = len(particles)
        # Checks
        if n<=1:
            print("WARNING: Cannot repulse 1 particle on its own, add more particles")
            return

        if repulsive_strat=="kernel":
            if n < phi_history_size:
                phi_history_size=n
            
            # Embed latent to smaller dimension
            model_input = particles
            with torch.no_grad():
                phi = model(model_input)

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
            # grads = jacrev(model)(particles)
            # grads = torch.autograd.functional.jacobian(
            #     model, 
            #     particles,
            # )

            # If batch too large
            grads = []
            for i in range(n):
                grad = torch.autograd.functional.jacobian(model, model_input[i].unsqueeze(0))
                grads.append(grad[0,:,0,...])
            grads = torch.stack(grads)
            
            # Set bandwidth of RBF kernel with median heuristic
            K.bandwidth(phi_history, phi_history)
            print(f"Sigma: {K.sigma.item()}")
            # Kernel grad shape: embedding_size x num_particles x num_particles
            kernel_grad = K.grad_first(phi_history, phi_history)
            kernel_grad_sum = torch.sum(kernel_grad, axis=-1)
        elif repulsive_strat=="score":
            score_sum = torch.sum(scores,dim=0)

        # Repulsive term TODO: vectorize this for loop
        new_particles = []
        for i in range(n):
            if repulsive_strat=="kernel":
                # Pushforward/Chain rule
                repulsive = torch.einsum('i,iklm->klm',kernel_grad_sum[:,i-n], grads[i,...]) / n
            elif repulsive_strat=="score":
                # Mean of other particle scores (TODO: could optimise this)
                repulsive = (score_sum - scores[i]) / (n-1)
                

            # (debug) Get repulsion norm
            # repulsive_norm = torch.norm(repulsive.reshape(repulsive.shape[0], -1), dim=-1).mean()
            # score_norm = torch.norm(scores[i].reshape(scores[i].shape[0], -1), dim=-1).mean()
            # repulsive_scale = snr * score_norm / repulsive_norm
            # print(repulsive_norm.item())

            # (debug) Save repulsion term
            # torch.save(repulsive, f'repulsive{i}.pt')

            # Score + Repulsion
            if repulse:
                new_particle = particles[i] + step_size * (scores[i] - repulsive_strength*repulsive)
            else:
                new_particle = particles[i] + step_size * (scores[i] + repulsive_strength*repulsive)
            # ONLY Repulsion
            # new_particle = particles[i] - step_size * (repulsive_scale * repulsive)
            if add_noise:
                noise = torch.randn(particles[i].shape, layout=particles[i].layout, device=device, generator=generator).to(device)
                new_particle += ((step_size * 2) ** 0.5) * noise

            new_particles.append(new_particle)
        return torch.stack(new_particles,axis=0)