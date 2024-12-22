import os
import random
import PIL.Image
from pathlib import Path
import pickle

import click
import torch
import dnnlib
import numpy as np

from gen_images import make_transform

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option("--images-per-person", type=int, default=1, help="Number of images to generate per person")
@click.option("--num-persons", type=int, default=1, help="Number of persons to generate images for")
def generate_image(
        network_pkl: str,
        outdir: str,
        images_per_person: int,
        num_persons: int) :
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = pickle.load(f)['G_ema'].cuda() # type: ignore

    os.makedirs(outdir, exist_ok=True)
    # Labels
    c = None

    for person_idx in range(num_persons):
        random_base = np.random.RandomState(person_idx).randn(1, G.z_dim)
        print(random_base)

        for dim in range(G.z_dim):
            for value in range(2):
                random_base = np.random.RandomState(person_idx).randn(1, G.z_dim)
                random_base[0][dim] = (value * 2 - 1) * 4
                z = torch.from_numpy(random_base).cuda()

                # Construct an inverse rotation/translation matrix and pass to the generator.  The
                # generator expects this matrix as an inverse to avoid potentially failing numerical
                # operations in the network.
                if hasattr(G.synthesis, 'input'):
                    translate = (0, 0)
                    rotate = 0
                    m = make_transform(translate, rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m))

                img = G(z, c, truncation_psi=0.7, noise_mode='const')
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                outdir_abs = Path(outdir).absolute()
                image_path = f'{outdir_abs}/person_{person_idx}_z_{dim}_{value}.png'
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(image_path)
                print(f"Generated image for person {person_idx} saved in '{image_path}'")

if __name__ == "__main__":
    generate_image()