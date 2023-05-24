"""
Embedding model for 64x64x4 latents of SD2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# import autograd.numpy as np

def init_weights(m):
    # Kaiming initialisation used in "A Random CNN Sees Objects" paper
    # TODO: THIS MIGHT NOT WORK
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

# CNN model
class CNN64(nn.Module):
    def __init__(self, relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # input is Nx4x64x64
        x = self.conv1(x) # Nx8x32x32
        x = self.act(x)
        x = self.conv2(x) # Nx8x16x16
        x = self.act(x)
        x = self.conv3(x) # Nx16x8x8
        x = self.act(x)
        x = self.conv4(x) # Nx16x4x4
        x = self.act(x)
        x = self.conv5(x) # Nx32x2x2
        x = self.act(x)
        x = self.conv6(x) # Nx64x1x1
        return x.squeeze(2,3)

class CNN16(nn.Module):
    def __init__(self, relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0)
        if relu:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # input is Nx4x64x64
        x = self.conv1(x) # Nx8x32x32
        x = self.act(x)
        x = self.conv2(x) # Nx8x16x16
        x = self.act(x)
        x = self.conv3(x) # Nx8x8x8
        x = self.act(x)
        x = self.conv4(x) # Nx16x4x4
        x = self.act(x)
        x = self.conv5(x) # Nx16x2x2
        x = self.act(x)
        x = self.conv6(x) # Nx16x1x1
        return x.squeeze(2,3)

class Average(nn.Module):
    def forward(self, x):
        return torch.mean(x, axis=(1,2,3)).unsqueeze(-1)

class AverageDim(nn.Module):
    def forward(self, x):
        return torch.mean(x, axis=(2,3))

class SoftBoundedAverage(nn.Module):
    def forward(self, x):
        # torch.tanh or torch.sigmoid
        return torch.tanh(torch.mean(x, axis=(1,2,3)).unsqueeze(-1))

class HardBoundedAverage(nn.Module):
    def forward(self, x):
        # TODO
        return torch.tanh(torch.mean(x, axis=(1,2,3)).unsqueeze(-1))

class VAEAverage(nn.Module):
    # VAE is too much gradient memory
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return torch.mean(image, axis=(2,3))

class Edges(nn.Module):
    def forward(self, x):
        dc = torch.mean(x, axis=(2,3))
        vert = torch.mean(x[...,32:, :] - x[...,:32,:], axis=(2,3))
        horz = torch.mean(x[...,:, :32] - x[...,:,32:], axis=(2,3))
        return torch.cat((dc, vert,horz),axis=1)

# Augmentations
class CombineKernel(nn.Module):
    # Add two kernels
    def __init__(self, k1, k2):
        # pass in two models
        super().__init__()
        self.k1 = k1
        self.k2 = k2
    
    def forward(self, x):
        return self.k1(x) + self.k2(x)
    
class Style(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, x):
        # Dot product similarity between filters (flattened upper triangular part of Gram matrix)
        filt = self.k(x) # NxF
        filt = filt.flatten(1)
        gram = torch.einsum('ni,nj->nij',filt,filt)

        # Flattened upper triangular 
        sims = []
        for i in range(len(x)):
            sim = gram[i][torch.triu_indices(*gram[0].shape).unbind()]
            sims.append(sim)
        return torch.stack(sims)


    

        