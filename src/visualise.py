## Functions to help visualise samples from diffusion

import torch
from PIL import Image

def image_grid(imgs, rows, cols):
    # assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def latent_to_img(latent_list):
    # Latent -> Visualisation
    # Convert latent unet output to format that can be visualised with plt.imshow
    res = []
    for latent in latent_list:
        lat = latent.permute(2,3,1,0).squeeze().cpu().detach()[...,:]
        lat -= lat.min()
        lat /= lat.max()
        res.append(lat)
    return res
def decode_latent(latent, model):
    # Latent -> VAE -> ImgOutput
    # scale and decode the image latents with vae
    scaling_factor = 0.18215 # pipe.vae.config.scaling_factor
    latents = 1 / scaling_factor * latent
    with torch.no_grad():
        image = model.decode(latents).sample
    return image
def output_to_img(out):
    # ImgOutput -> Visualisation
    # Convert vae decoded output image to format to be plotted
    image = (out / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    return image