"""
Embedding model for 64x64x4 latents of SD2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as np

def init_weights(m):
    # Kaiming initialisation used in "A Random CNN Sees Objects" paper
    if isinstance(m, nn.Linear):
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
        # input is 1x4x64x64
        x = self.conv1(x) # 1x8x32x32
        x = self.act(x)
        x = self.conv2(x) # 1x8x16x16
        x = self.act(x)
        x = self.conv3(x) # 1x16x8x8
        x = self.act(x)
        x = self.conv4(x) # 1x16x4x4
        x = self.act(x)
        x = self.conv5(x) # 1x32x2x2
        x = self.act(x)
        x = self.conv6(x) # 1x64x1x1
        return x.squeeze()

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
        # input is 1x4x64x64
        x = self.conv1(x) # 1x8x32x32
        x = self.act(x)
        x = self.conv2(x) # 1x8x16x16
        x = self.act(x)
        x = self.conv3(x) # 1x8x8x8
        x = self.act(x)
        x = self.conv4(x) # 1x16x4x4
        x = self.act(x)
        x = self.conv5(x) # 1x16x2x2
        x = self.act(x)
        x = self.conv6(x) # 1x16x1x1
        return x.squeeze()

class Average(nn.Module):
    def forward(self, x):
        return torch.mean(x).reshape(1)

class SoftBoundedAverage(nn.Module):
    def forward(self, x):
        # torch.tanh or torch.sigmoid
        return torch.tanh(torch.mean(x)).reshape(1)

class HardBoundedAverage(nn.Module):
    def forward(self, x):
        # TODO
        return torch.tanh(torch.mean(x)).reshape(1)

class VAEAverage(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
    
    def forward(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.mean(axis=(0,2,3))
