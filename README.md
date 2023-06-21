Using score_sde_pytorch folder virtual environment

# Guide
Use -h on scripts to find out what 

## Repulsion API
Repulsion is done in repulse.py

Run:
python repulse.py --method repulsive --model averagedim --strength 100

The number of particles and prompt can be defined within the script.

## Experimentation notebooks
Much of the exploration of sampling techniques was done in the following Python notebooks:


## Style classifier

## Report experiments

data_generator.py
- Generate each dataset for evaluation specified by --mode

# Glossary
## Repulsion methods
averagedim: latent channel average method
cnn16: random CNN method (with 16 dimensional output embedding)
ro3: rule of thirds
vgg_noise: style classifier (vgg model with conditioning on the noise level of latent)
vgg_noisero3: style classifier with rule of thirds



## Evaluation
The FID score is computed on the generated images using the [the PyTorch port](https://github.com/mseitzer/pytorch-fid).

