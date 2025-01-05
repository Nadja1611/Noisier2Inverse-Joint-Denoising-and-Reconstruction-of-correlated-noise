import os
import LION.CTtools.ct_utils as ct
import LION.CTtools.ct_geometry as ctgeo
import numpy as np
import torch
from tomosipo.torch_support import (
    to_autograd,
)



def is_image_file(filename):
    extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.tif', '.TIF']
    return any(filename.endswith(extension) for extension in extensions)

def make_dataset(dir):
    img_paths = []
    assert os.path.isdir(dir), '{} is not a valid directory'.format(dir)

    for (root, dirs, files) in sorted(os.walk(dir)):
        for filename in files:
            if is_image_file(filename):
                img_paths.append(os.path.join(root, filename))
    return img_paths



# %%

photon_count=100   # 
attenuation_factor=2.76 # corresponds to absorption of 50%
def apply_noise(img, photon_count):
    opt = dict(dtype=np.float32)
    img = np.exp(-img, **opt)
    # Add poisson noise and retain scale by dividing by photon_count
    img = np.random.poisson(img * photon_count)
    img[img == 0] = 1
    img = img / photon_count
    # Redo log transform and scale img to range [0, img_max] +- some noise.
    img = -np.log(img, **opt)
    return img

''' this function adds gaussian noise to our sinograms '''
def add_gaussian_noise(img, sigma):
    img = np.array(img)
    noise = np.random.normal(0, sigma, img.shape)/100
    img = img + np.max(img)*noise
    return torch.tensor(img)


def create_noisy_sinograms(images, angles_full, sigma):
    # 0.1: Make geometry:
    geo = ctgeo.Geometry.parallel_default_parameters(
        image_shape=images.shape, number_of_angles=angles_full
    )  # parallel beam standard CT
    # 0.2: create operator:
    op = ct.make_operator(geo)
    # 0.3: forward project:
    sino = op(torch.from_numpy(images))
    sinogram_full = add_gaussian_noise(sino, sigma)
    sinogram_full = torch.moveaxis(sinogram_full, -1, -2)
    return np.asarray(sinogram_full.unsqueeze(1))
