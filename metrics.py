"""Plot the closest and furthest images in each set according to l2 distance between features"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
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
from src.score_utils import get_sigmas, get_score_input, scale_input, get_score, step_score
from src.denoise_utils import denoise
from src.sampling_utils import random_step, langevin_step, repulsive_step_parallel
from src.kernel import RBF
from src.embedding import CNN64, CNN16, init_weights, AverageDim, Average, VAEAverage, Style, VGG, VGG_dropout, VGG_noise
from src.datasets import StyleDataset, ArtistDataset
from src.train import train
from src.metric_utils import fid_metric, get_metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob

# prompts = ['vase','tree','parkour','cave','kite']
prompts = ['cave']
exp_names = {
    'max_div': 'max diversity',
    'min_div':'minimum diversity',
    'averagedim_all_r1000':'average channel repulsed',
    'cnn16_all_r1000':'random CNN repulsed',
    'langevin': 'langevin',
    'ro3_all_r1000': 'rule of thirds repulsed',
    'vgg_noise_all_r1000':'style classifier repulsed',
    'vgg_noisero3_all_r1000':'style classifier and rule of thirds repulsed'
}
met_names = {
    'stats': 'FID',
    'gram64': 'Style',
    'local': 'Location'
}
for prompt in prompts:
    os.makedirs(f'data/final/{prompt}/plots/', exist_ok=True)
    for exp in exp_names:
        # if prompt=='vase' and (exp=='cnn16_all_r1000' or exp=='max_div'):
        #     continue
        for met in met_names:
            f = []
            if exp in ['max_div','max_div2']:
                for file in glob(f"data/final/{prompt}/{exp}/*/*.png"):
                    f.append(file)
            else:
                for file in glob(f"data/final/{prompt}/{exp}/all/*.png"):
                    f.append(file)
            # Get metric features
            act = get_metric(f, met, return_act=True)
            # Get sorted distance pairs
            dists = {}
            n = len(act)
            for i in range(n):
                for j in range(i+1,n):
                    d = np.linalg.norm(act[i]-act[j])
                    dists[i,j] = d
            dists_sorted = sorted(dists.items(), key=lambda x:x[1])

            
            n=5
            n_rows=2
            n_cols=n*2
            wspace = 0.6
            fig = plt.figure(figsize=(10,1.6), dpi=300)
            for i in range(n):
                # Most similar
                p_i, p_j = dists_sorted[i][0]
                # print(dists_sorted[i][1])
                f_i = f[p_i]
                f_j = f[p_j]
                ax = plt.subplot(n_rows,n_cols,i+1)
                plt.imshow(plt.imread(f_i))
                plt.subplots_adjust(hspace=0.1, wspace=wspace)
                plt.xticks([])
                plt.yticks([])
                ax = plt.subplot(n_rows,n_cols,n_cols+i+1)
                plt.imshow(plt.imread(f_j))
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(hspace=0.1, wspace=wspace)
                plt.xlabel(str(round(dists_sorted[i][1],2)))
            for i in range(-n,0):
                # Least similar
                p_i, p_j = dists_sorted[i][0]
                # print(dists_sorted[-1-i][1])
                f_i = f[p_i]
                f_j = f[p_j]
                ax = plt.subplot(n_rows,n_cols,i+2*n+1)
                plt.imshow(plt.imread(f_i))
                plt.subplots_adjust(hspace=0.1, wspace=wspace)
                plt.xticks([])
                plt.yticks([])
                ax = plt.subplot(n_rows,n_cols,n_cols+i+2*n+1)
                plt.imshow(plt.imread(f_j))
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(hspace=0.1, wspace=wspace)
                plt.xlabel(str(round(dists_sorted[i][1],2)))

            plt.suptitle(f'{met_names[met]} feature L2 distance pairs - {exp_names[exp]} set')
            plt.show()
            dst = f'data/final/{prompt}/plots/{exp}_{met}_dists.pdf'
            plt.savefig(dst)