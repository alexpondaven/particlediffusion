"""
Embedding model for 64x64x4 latents of SD2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import autograd.numpy as np

# CNN model
class CNN(nn.Module):
    # Resnet50 inspired
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.act = nn.ReLU()

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

def init_weights(m):
    # Kaiming initialisation used in "A Random CNN Sees Objects" paper
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

