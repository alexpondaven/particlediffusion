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

# Trained model
class VGG(nn.Module):
    def __init__(self, num_outputs=5, logsoftmax=True, return_conv_act=False, return_pre_fconv=False):
        super().__init__()

        self.num_outputs = num_outputs
        self.logsoftmax = logsoftmax
        self.return_conv_act = return_conv_act
        self.return_pre_fconv = return_pre_fconv

        self.conv_layers = []
        self.num_blocks = 4
        for i in range(self.num_blocks):
            layer = nn.Conv2d(4*2**i,4*2**(i+1),kernel_size=(3,3), stride=1, padding=1)
            self.conv_layers.append(layer)
        final_nchannels = 4*2**self.num_blocks
        conv_final = nn.Conv2d(final_nchannels, final_nchannels, kernel_size=(4,4))
        self.conv_layers.append(conv_final)

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.fc = nn.Linear(in_features=final_nchannels, out_features=num_outputs, bias=True)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i != len(self.conv_layers)-1:
                x = self.maxpool(self.act(x))
            
            # Return pre-final conv layer
            if i==(len(self.conv_layers)-2) and self.return_pre_fconv:
                return x

        
        x = x.squeeze(-2,-1)
        if self.return_conv_act:
            return x
        # FC layer
        x = self.fc(x)
        if self.logsoftmax:
            x = self.logsoftmax(x)
        
        return x

class VGG_noise(nn.Module):
    def __init__(self, num_outputs=5, logsoftmax=True, return_conv_act=False, return_pre_fconv=False):
        super().__init__()

        self.num_outputs = num_outputs
        self.logsoftmax = logsoftmax
        self.return_conv_act = return_conv_act
        self.return_pre_fconv = return_pre_fconv

        self.conv_layers = []
        self.num_blocks = 4
        for i in range(self.num_blocks):
            layer = nn.Conv2d(4*2**i,4*2**(i+1),kernel_size=(3,3), stride=1, padding=1)
            self.conv_layers.append(layer)
        final_nchannels = 4*2**self.num_blocks
        conv_final = nn.Conv2d(final_nchannels, final_nchannels, kernel_size=(4,4))
        self.conv_layers.append(conv_final)

        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.fc = nn.Linear(in_features=final_nchannels + 1, out_features=num_outputs, bias=True)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lvl_idx):
        # Conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i != len(self.conv_layers)-1:
                x = self.maxpool(self.act(x))
            
            # Return pre-final conv layer
            if i==(len(self.conv_layers)-2) and self.return_pre_fconv:
                return x
        
        x = x.squeeze(-2,-1)
        if self.return_conv_act:
            return x
        
        # FC layer - input Nx64
        # Add on noise_level sigma
        # 1) Add as extra dimension
        # 2) Input to hypernetwork for FC layer weights
        x = torch.cat([x, lvl_idx.unsqueeze(1)],dim=1)
        x = self.fc(x)
        if self.logsoftmax:
            x = self.logsoftmax(x)
        
        return x

# Adding dropout didn't help much for overfitting
class VGG_dropout(nn.Module):
    def __init__(self, num_outputs=5, logsoftmax=True, return_conv_act=False):
        super().__init__()

        self.num_outputs = num_outputs
        self.logsoftmax = logsoftmax
        self.return_conv_act = return_conv_act

        self.conv_layers = []
        self.bn_layers = []
        self.num_blocks = 5
        for i in range(self.num_blocks):
            layer = nn.Conv2d(4*2**i,4*2**(i+1),kernel_size=(3,3), stride=1, padding=1)
            bn = nn.BatchNorm2d(4*2**(i+1))
            self.conv_layers.append(layer)
            self.bn_layers.append(bn)
        final_nchannels = 4*2**self.num_blocks
        conv_final = nn.Conv2d(final_nchannels, final_nchannels, kernel_size=(2,2))
        self.conv_layers.append(conv_final)

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.d1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(final_nchannels, final_nchannels//2)
        self.d2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(final_nchannels//2, num_outputs)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i != len(self.conv_layers)-1:
                x = self.bn_layers[i](x)
                x = self.maxpool(self.act(x))
        
        x = x.squeeze(-2,-1)
        if self.return_conv_act:
            return x
        # FC layer
        x = self.fc1(self.d1(x))
        x = self.act(x)
        x = self.fc2(self.d2(x))
        if self.logsoftmax:
            x = self.logsoftmax(x)
        
        return x

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

class Latent(nn.Module):
    def forward(self, x):
        return x.flatten(1)
    
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
        # dc = torch.mean(x, axis=(2,3))
        horz = torch.mean(x[...,32:, :] - x[...,:32,:], axis=(2,3))
        vert = torch.mean(x[...,:, :32] - x[...,:,32:], axis=(2,3))
        # return vert
        # return torch.cat((dc, vert,horz),axis=1)
        return torch.cat((vert,horz),axis=1)

# Rule of thirds
class RuleOfThirds(nn.Module):
    def forward(self, x):
        # dc = torch.mean(x, axis=(2,3))
        h1 = torch.mean(x[...,:21, :] - x[...,21:42,:], axis=(2,3))
        h2 = torch.mean(x[...,22:43, :] - x[...,43:,:], axis=(2,3))
        v1 = torch.mean(x[...,:,:21] - x[...,:,21:42], axis=(2,3))
        v2 = torch.mean(x[...,:,22:43] - x[...,:,43:], axis=(2,3))
        # return vert
        return torch.cat((h1, h2, v1,v2),axis=1)

# Rule of thirds on VGG features
class VGGRo3(nn.Module):
    def __init__(self, vgg, mode="ro3", noise_cond=False):
        super().__init__()
        self.vgg = vgg
        self.mode = mode
        self.noise_cond = noise_cond

    def forward(self, x, lvl=None):
        # Nx64x4x4
        if self.noise_cond:
            x = self.vgg(x, lvl)
        else:
            x = self.vgg(x)
        
        if self.mode=="dc_average":
            ## DC over 4x4 spatial
            dc_spatial = torch.mean(x, axis=(2,3))
            return dc_spatial
        elif self.mode=="dc_channel":
            ## DC over channels, then flatten
            dc_channel = torch.mean(x, axis=1).flatten(1)
            return dc_channel
        elif self.mode=="edge_channel":
            ## Difference in spatial (mean over channels)
            h_channel = torch.mean(x[...,:2,:] - x[...,2:,:], axis=1).flatten(1)
            v_channel = torch.mean(x[...,:,:2] - x[...,:,2:], axis=1).flatten(1)
            edge_channel = torch.cat((h_channel,v_channel),axis=1)
            return edge_channel
        elif self.mode=="edge_spatial":
            ## Difference in spatial (average over 2x4 spatial) <- makes more sense, mean difference over left/right
            h_spatial = torch.mean(x[...,:2,:] - x[...,2:,:], axis=(2,3))
            v_spatial = torch.mean(x[...,:,:2] - x[...,:,2:], axis=(2,3))
            edge_spatial = torch.cat((h_spatial,v_spatial),axis=1)
            return edge_spatial
        elif self.mode=="ro3":
            ## RO3 Spatial
            h1 = torch.mean(x[...,0, :] - x[...,1,:], axis=2)
            h2 = torch.mean(x[...,2, :] - x[...,3,:], axis=2)
            v1 = torch.mean(x[...,:,0] - x[...,:,1], axis=2)
            v2 = torch.mean(x[...,:,2] - x[...,:,3], axis=2)
            ro3_spatial = torch.cat((h1,h2,v1,v2),axis=1)
            return ro3_spatial
        else:
            print("ERROR: Not valid VGG operation mode")
            return x

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


    

        