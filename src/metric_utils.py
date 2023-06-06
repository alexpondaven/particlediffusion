# Functions to apply on sets of generated samples
from fid.fid_score import calculate_activation_statistics, save_statistics, load_statistics, calculate_frechet_distance, get_activations
from fid.inception import InceptionV3
import os
from glob import glob
import numpy as np

## Metrics
def get_metric(f, metric):
    # Compute metric on f filenames
    fid_dir = "fid" #/pt_inception-2015-12-05-6726825d.pth"
    device = 'cuda'
    if metric=='stats':
        dims = 2048  # 64, 192, 768, 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx], model_dir=fid_dir).to(device)
        mu, sigma = calculate_activation_statistics(f, model, 50, dims, device)
    elif metric=='local':
        dims = 192  # 64, 192, 768, 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx], model_dir=fid_dir).to(device)

        act = get_activations(f, model, 50, dims, device, None)
        act = np.mean(act,axis=1)
        act = act.reshape(act.shape[0], -1)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
    elif metric=='gram64':
        dims = 64  # 64, 192, 768, 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx], model_dir=fid_dir).to(device)
        
        act = get_activations(f, model, 50, dims, device, None, gap=True)
        gram = np.einsum('ni,nj->nij',act,act)
        # Flattened upper triangular 
        sims = []
        for i in range(len(gram)):
            sim = gram[i][np.triu_indices(dims)]
            sims.append(sim)
        sims = np.stack(sims)
        mu = np.mean(sims, axis=0)
        sigma = np.cov(sims, rowvar=False)
    return mu, sigma

def fid_metric(exp, metrics=['stats','local','gram64'], subject='cave', artistnum=-1):
    if artistnum==-1:
        foldername="base"
    else:
        foldername=f"artist_{artistnum}"
    # Calculate fid metrics
    fid_metrics = {}
    for metric in metrics:
        max_div_path = f"data/final/{subject}/max_div_{metric}.npz"
        metrics_folder = f"data/final/{subject}/metrics_{foldername}"
        if metric=="max_div2":
            exp_path = f"data/final/{subject}/max_div2_{metric}.npz"
        else:
            exp_path = f"{metrics_folder}/{exp}_{metric}.npz"
            os.makedirs(metrics_folder, exist_ok=True)
        
        # Check if max_div exists
        if os.path.exists(max_div_path):
            m0, s0 = load_statistics(max_div_path)
        else:
            # Read image sets
            f = []
            for file in glob(f"data/final/{subject}/max_div/*/*.png"):
                f.append(file)
            m0,s0 = get_metric(f, metric)
            save_statistics(max_div_path,m0,s0)
        
        if os.path.exists(exp_path):
            m,s = load_statistics(exp_path)
        else:
            f = []
            if metric=="max_div2":
                subset="*"
            elif artistnum==-1:
                subset="base"
            else:
                ### TODO: CHOOSE ARTIST CORRESPONDING TO ARTISTNUM
                subset="Thomas_Kinkade"
            for file in glob(f"data/final/{subject}/{exp}/{subset}/*.png"):
                f.append(file)
            m, s = get_metric(f, metric)
            save_statistics(exp_path,m,s)
        
        fid = calculate_frechet_distance(m0, s0, m, s)
        fid_metrics[metric] = fid
    return fid_metrics

def save_features(f):
    # Save InceptionNet features 2048
    fid_dir = "fid" #/pt_inception-2015-12-05-6726825d.pth"
    dims = 2048  # 64, 192, 768, 2048
    device = 'cuda'
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=fid_dir).to(device)
    m, s = calculate_activation_statistics(f, model, 50, dims, device)
    act = get_activations(f, model, 50, dims, device, None)