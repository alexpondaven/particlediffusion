# Train style classification model on given latent data
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from src.embedding import CNN64, CNN16, init_weights, AverageDim, Average, VAEAverage, Style, VGG, VGG_dropout, VGG_noise
from src.datasets import StyleDataset, ArtistDataset
from src.train import train

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob

# Dataset
# dataset = StyleDataset()
dataset = ArtistDataset(return_scores=False)
train_sz = int(0.7 * len(dataset))
val_sz = int(0.1 * len(dataset))
test_sz = len(dataset) - train_sz - val_sz
train_set, val_set, test_set = random_split(dataset, [train_sz,val_sz,test_sz])

# Dataloader
batch_size = 32
train_dl = DataLoader(train_set, batch_size=32, shuffle=True)
val_dl = DataLoader(val_set, batch_size=val_sz)
test_dl = DataLoader(test_set, batch_size=test_sz)

# Model training
model = VGG_noise(num_outputs=len(dataset.styles))
device="cuda"
model.to(device)

criterion = nn.NLLLoss(reduction='none')
lr = 1e-5
epochs = 2000
opt = optim.Adam(model.parameters(), lr=lr)
batches = len(train_dl)

seed=0
generator = torch.Generator(device).manual_seed(seed)
torch.manual_seed(seed)
train_losses, val_losses, train_accs, val_accs = train(model, criterion, epochs, opt, train_dl, val_dl, dataset.noise_levels, score=True)