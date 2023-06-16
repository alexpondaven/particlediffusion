"""Plot the closest and furthest images in each set according to l2 distance between features"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import matplotlib.pyplot as plt

import pickle
from src.metric_utils import fid_metric, get_metric
from glob import glob

# prompts = ['vase','tree','parkour','cave','kite','actress']
prompts = ['vase']
exp_names = {
    'max_div': 'max diversity',
    'min_div':'minimum diversity',
    'averagedim_all_r1000':'average channel repulsed',
    'cnn16_all_r1000':'random CNN repulsed',
    'langevin': 'langevin',
    # 'eulera': 'stochastic scheduler',
    'ro3_all_r1000': 'rule of thirds repulsed',
    'vgg_noise_all_r1000':'style classifier repulsed',
    'vgg_noisero3_all_r1000':'style classifier and rule of thirds repulsed'
}
met_names = {
    'stats': 'FID',
    'gram64': 'Style',
    'local': 'Location'
}

plot_dists = False
plot_hists = False

for prompt in prompts:
    os.makedirs(f'data/final/{prompt}/plots/', exist_ok=True)
    for exp in exp_names:
        for met in met_names:
            act_path = f"data/final/{prompt}/metrics_all/{exp}_{met}_act.npz"
            dists_path = f"data/final/{prompt}/metrics_all/{exp}_{met}_dists.pkl"
            f = []
            if exp in ['max_div','max_div2']:
                for file in glob(f"data/final/{prompt}/{exp}/*/*.png"):
                    f.append(file)
            else:
                for file in glob(f"data/final/{prompt}/{exp}/all/*.png"):
                    f.append(file)
            # Get metric features
            if os.path.exists(act_path):
                act = np.load(act_path)['act']
            else:
                act = get_metric(f, met, return_act=True)
                np.savez(act_path, act=act)
            # Get sorted distance pairs
            if os.path.exists(dists_path):
                with open(dists_path,'rb') as file: 
                    dists_sorted = pickle.load(file)
            else:
                dists = {}
                n = len(act)
                for i in range(n):
                    for j in range(i+1,n):
                        d = np.linalg.norm(act[i]-act[j])
                        dists[i,j] = d
                dists_sorted = sorted(dists.items(), key=lambda x:x[1])
                with open(dists_path,'wb') as file: 
                    pickle.dump(dists_sorted, file)
            
            if plot_hists:
                dists_only = [d[1] for d in dists_sorted]
                plt.figure()
                plt.hist(dists_only, bins=50)
                plt.title(f'Distance histogram - {exp_names[exp]} set')
                plt.xlabel(f'Euclidean distance between {met_names[met]} features')
                plt.ylabel('Frequency')
                dst = f'data/final/{prompt}/plots/{exp}_{met}_hist.pdf'
                plt.savefig(dst)
            
            #### Images with smallest and largest distance
            if plot_dists:
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