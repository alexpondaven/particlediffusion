import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from diffusers import DiffusionPipeline
import numpy as np
import random
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import pandas as pd

from fid.fid_score import calculate_activation_statistics, save_statistics, load_statistics, calculate_frechet_distance, get_activations
from fid.inception import InceptionV3

from PIL import Image
from tqdm.auto import tqdm
from torch import autocast

from src.visualise import image_grid, latent_to_img, decode_latent, output_to_img
from schedulers.euler_discrete import EulerDiscreteCustomScheduler, FrozenDict, randn_tensor
from src.score_utils import get_sigmas, get_score_input, scale_input, get_score, step_score, denoise
from src.sampling_utils import random_step, langevin_step, repulsive_step_parallel
from src.kernel import RBF
from src.embedding import CNN64, CNN16, init_weights, AverageDim, Average, VAEAverage, Style, VGG, VGG_dropout
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
dataset = ArtistDataset(return_scores=True)
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
model = VGG(num_outputs=len(dataset.styles))
device="cuda"
model.to(device)

criterion = nn.NLLLoss()
lr = 1e-4
epochs = 100
opt = optim.Adam(model.parameters(), lr=lr)
batches = len(train_dl)

seed=0
generator = torch.Generator(device).manual_seed(seed)
torch.manual_seed(seed)
train_losses, val_losses, train_accs, val_accs = train(model, criterion, epochs, opt, train_dl, val_dl, dataset.noise_levels)