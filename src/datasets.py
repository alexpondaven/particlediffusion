from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
import pandas as pd
import torch
import os

class StyleDataset(Dataset):
    def __init__(self, device="cuda"):
        self.img_dir = 'data/styles'
        self.noise_levels = 21
        self.device = device

        styles = pd.read_csv(f"{self.img_dir}/styles.csv",header=None)
        self.styles = [style.replace(" ", "_") for style in styles[0]]
        self.styles_num = []
        for style in self.styles:
            self.styles_num.append(len(glob(f'{self.img_dir}/{style}/*')))
        
        # Label format: (style, #data, noise_levels)
        # self.labels = [(label, n, noise_lvl) for label in range(len(self.styles)) for n in range(self.styles_num[label]) for noise_lvl in range(21)]
        self.labels = [(label, n) for label in range(len(self.styles)) for n in range(self.styles_num[label])]
        
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Returns Nx4x64x64 latent, 
        # Choose style_i, #data, noise_levels
        # label, n, noise_lvl = self.labels[idx]
        label, n = self.labels[idx]
        style = self.styles[label]
        img_path = os.path.join(self.img_dir, style, f"tree_{style}_{n}.pt")
        latents = torch.load(img_path)
        x = latents.to(self.device)
        y = torch.tensor(label).to(self.device)
        return x,y

class ArtistDataset(Dataset):
    def __init__(self, device="cuda", return_scores=False):
        self.img_dir = 'data/styles'
        self.noise_levels = 21
        if return_scores:
            # Score describes transition between latents, so there is one less
            self.noise_levels -= 1
        self.device = device
        self.mode = "latent"
        if return_scores:
            self.mode = "score"

        styles = pd.read_csv(f"{self.img_dir}/artists.csv",header=None)
        self.styles = [style.replace(" ", "_") for style in styles[0]]
        f = open("data/styles/base_prompt.csv",'r')
        subjects = f.readlines()
        self.styles_num = len(subjects)
        
        # Label format: (style, #data)
        self.labels = [(label, n) for label in range(len(self.styles)) for n in range(self.styles_num)]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Returns Nx4x64x64 latent, 
        # Choose style_i, #data, noise_levels
        label, n = self.labels[idx]
        style = self.styles[label]
        img_path = os.path.join(self.img_dir, "artists", style, f"artist{label}_subject{n}_{self.mode}.pt")
        latents = torch.load(img_path)
        x = latents.to(self.device)
        y = torch.tensor(label).to(self.device)
        return x,y