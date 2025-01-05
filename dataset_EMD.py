import os
import numpy as np
import cv2 as cv
from skimage.transform import rescale, resize
import torch
from torch.utils.data import TensorDataset, DataLoader
from tomosipo.torch_support import (
    to_autograd,
)
from torchvision import transforms
import gc
import matplotlib.pyplot as plt
import LION.CTtools.ct_geometry as ctgeo
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils_inverse import *



#%% function for reading in our walnut data
def get_images(path, amount_of_images='all', scale_number=1):
    all_images = []
    all_image_names = os.listdir(path)
    if amount_of_images == 'all':
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    
    return all_images


class Walnut(Dataset):
    def __init__(self, data_dir=None, noise_type='salt_and_pepper', noise_intensity=0.05, noise_sigma=2, train=True, transform=None):
        super(Walnut, self).__init__()

        self.noise_intensity = noise_intensity
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type

 
        if train:
            self.clean_dir = os.path.join(data_dir, 'train')
        else:
            self.clean_dir = os.path.join(data_dir, 'test')


        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
                    #### add salt and peppern noise to given images 
    def add_salt_and_pepper_noise(self, image, salt_ratio, pepper_ratio): 
        """
        Adds salt and pepper noise to an image.

        Args:
            image (numpy.ndarray): Input image.
            salt_ratio (float): Ratio of salt noise (default: 0.05).
            pepper_ratio (float): Ratio of pepper noise (default: 0.05).

        Returns:
            numpy.ndarray: Image with salt and pepper noise.
        """
        pepper_ratio=self.noise_intensity
        salt_ratio=self.noise_intensity
        row, col = image.shape
        salt = np.random.rand(row, col) < salt_ratio
        pepper = np.random.rand(row, col) < pepper_ratio
        noisy_image = np.copy(image)
        noisy_image[salt] = 1
        noisy_image[pepper] = 0
        return noisy_image        
    
    def normalize(self, image):
        for i in range(len(image)):
            image[i] = image[i] - np.min(image[i])
            image[i] = image[i]/((np.max(image[i])+1e-5))
        return image   

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        if clean_path.endswith('.png'):
            clean = np.array(cv.imread(clean_path, cv.IMREAD_GRAYSCALE),dtype=np.float16)
        elif clean_path.endswith('.pt'):
            clean = torch.load(clean_path)
            cleani = np.zeros_like(clean)
            cleani[80:-80, 45:-45] = clean[80:-80, 45:-45]
            clean = cleani
        
        if self.noise_type == 'gauss':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean /= np.max(clean, axis = (-1,-2))
            clean_sino = np.array((create_noisy_sinograms(np.expand_dims(clean,0), 512, 0)).squeeze(0))
            noisy = np.asarray(add_correlated_noise(clean_sino, self.noise_intensity, self.noise_sigma))
            noisier = np.asarray(add_correlated_noise(noisy, self.noise_intensity, self.noise_sigma))
            noise = clean_sino - noisy
            #del(clean_sino)

        elif self.noise_type == 'gauss_image':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean /= np.max(clean, axis = (-1,-2))
            clean_new = np.copy(clean)
            noisy_image = np.asarray(add_gaussian_noise(clean_new, self.noise_intensity))
            noisier_image = np.asarray(add_gaussian_noise(noisy_image, self.noise_intensity))
            noisy = np.array((create_noisy_sinograms(np.expand_dims(noisy_image,0), 512, 0)).squeeze(0))
            noisier = np.array((create_noisy_sinograms(np.expand_dims(noisier_image,0), 512, 0)).squeeze(0))
            del(noisy_image, clean_new)

        elif self.noise_type == 'poisson':
            noisy = np.random.poisson(clean * 1 * self.noise_intensity) / self.noise_intensity / 1.
            # Add Poisson
            noisier = noisy + (np.random.poisson(clean * 1 * self.noise_intensity) / self.noise_intensity / 255. - clean)

        elif self.noise_type == 'salt_and_pepper':
            #clean = self.normalize(clean)
            noisy = self.add_salt_and_pepper_noise(clean, salt_ratio= self.noise_intensity, pepper_ratio=self.noise_intensity)
            noisier = self.add_salt_and_pepper_noise(noisy, salt_ratio= self.noise_intensity, pepper_ratio=self.noise_intensity )
        else:
            raise NotImplementedError('wrong type of noise')
        if self.noise_type == "salt_and_pepper":
            clean, noisy, noisier = self.transform(clean), self.transform(noisy), self.transform(noisier)
            clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        elif self.noise_type == "gauss" or self.noise_type == 'gauss_image':
            clean, noisy, noisier = self.transform(clean),  self.transform(noisy), self.transform(noisier)
            clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        else:
            raise NotImplementedError('wrong type of noise')
        if self.noise_type == "salt_and_pepper":
            return {'clean': clean, 'noisy': noisy, 'noisier': noisier}
        elif self.noise_type == "gauss" or self.noise_type == "gauss_image":
            return {'clean': clean, 'noisy': noisy, 'noisier': noisier, 'noise': noise}
            del(clean, noisy, noisier, noise)
        else:
            raise NotImplementedError('wrong type of noise')
    def __len__(self):
        return len(self.clean_paths)





class Walnut_test(Dataset):
    def __init__(self, data_dir=None, noise_type='gauss', noise_sigma = 2,  noise_intensity=1.0, transform=None):
        super(Walnut, self).__init__()

        self.noise_intensity = noise_intensity
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type

        self.clean_dir = os.path.join(data_dir, 'test_paper')


        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
                    #### add salt and peppern noise to given images 
    def add_salt_and_pepper_noise(self, image, salt_ratio, pepper_ratio): 
        """
        Adds salt and pepper noise to an image.

        Args:
            image (numpy.ndarray): Input image.
            salt_ratio (float): Ratio of salt noise (default: 0.05).
            pepper_ratio (float): Ratio of pepper noise (default: 0.05).

        Returns:
            numpy.ndarray: Image with salt and pepper noise.
        """
        pepper_ratio=self.noise_intensity
        salt_ratio=self.noise_intensity
        row, col = image.shape
        salt = np.random.rand(row, col) < salt_ratio
        pepper = np.random.rand(row, col) < pepper_ratio
        noisy_image = np.copy(image)
        noisy_image[salt] = 1
        noisy_image[pepper] = 0
        return noisy_image        
    
    def normalize(self, image):
        for i in range(len(image)):
            image[i] = image[i] - np.min(image[i])
            image[i] = image[i]/((np.max(image[i])+1e-5))
        return image   

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        if clean_path.endswith('.png'):
            clean = np.array(cv.imread(clean_path, cv.IMREAD_GRAYSCALE),dtype=np.float16)
        elif clean_path.endswith('.pt'):
            clean = torch.load(clean_path)
            cleani = np.zeros_like(clean)
            cleani[80:-80, 45:-45] = clean[80:-80, 45:-45]
            clean = cleani
        
        if self.noise_type == 'gauss':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean /= np.max(clean, axis = (-1,-2))
            clean_sino = np.array((create_noisy_sinograms(np.expand_dims(clean,0), 512, 0)).squeeze(0))
            noisy = np.asarray(add_correlated_noise(clean_sino, self.noise_intensity, self.noise_sigma))
            noisier = np.asarray(add_correlated_noise(noisy, self.noise_intensity, self.noise_sigma))
            noise = clean_sino - noisy
            del(clean_sino)

        elif self.noise_type == 'gauss_image':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean /= np.max(clean, axis = (-1,-2))
            clean_new = np.copy(clean)
            noisy_image = np.asarray(add_gaussian_noise(clean_new, self.noise_intensity))
            noisier_image = np.asarray(add_gaussian_noise(noisy_image, self.noise_intensity))
            noisy = np.array((create_noisy_sinograms(np.expand_dims(noisy_image,0), 512, 0)).squeeze(0))
            noisier = np.array((create_noisy_sinograms(np.expand_dims(noisier_image,0), 512, 0)).squeeze(0))
            del(noisy_image, clean_new)

        elif self.noise_type == 'poisson':
            noisy = np.random.poisson(clean * 1 * self.noise_intensity) / self.noise_intensity / 1.
            # Add Poisson
            noisier = noisy + (np.random.poisson(clean * 1 * self.noise_intensity) / self.noise_intensity / 255. - clean)

        elif self.noise_type == 'salt_and_pepper':
            #clean = self.normalize(clean)
            noisy = self.add_salt_and_pepper_noise(clean, salt_ratio= self.noise_intensity, pepper_ratio=self.noise_intensity)
            noisier = self.add_salt_and_pepper_noise(noisy, salt_ratio= self.noise_intensity, pepper_ratio=self.noise_intensity )
        else:
            raise NotImplementedError('wrong type of noise')
        if self.noise_type == "salt_and_pepper":
            clean, noisy, noisier = self.transform(clean), self.transform(noisy), self.transform(noisier)
            clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        elif self.noise_type == "gauss" or self.noise_type == 'gauss_image':
            clean, noisy, noisier = self.transform(clean),  self.transform(noisy), self.transform(noisier)
            clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        else:
            raise NotImplementedError('wrong type of noise')
        if self.noise_type == "salt_and_pepper":
            return {'clean': clean, 'noisy': noisy, 'noisier': noisier}
        elif self.noise_type == "gauss" or self.noise_type == "gauss_image":
            return {'clean': clean, 'noisy': noisy, 'noisier': noisier, 'noise': noise}
            del(clean, noisy, noisier, noise)
        else:
            raise NotImplementedError('wrong type of noise')
    def __len__(self):
        return len(self.clean_paths)





class Walnut_test(Dataset):
    def __init__(self, data_dir=None, noise_type='salt_and_pepper', noise_sigma = 2,  noise_intensity=1.0, transform=None):
        super(Walnut_test, self).__init__()

        self.noise_intensity = noise_intensity
        self.noise_sigma = noise_sigma
        self.noise_type = noise_type

        self.clean_dir = os.path.join(data_dir, 'test_paper')


        self.clean_paths = sorted(make_dataset(self.clean_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
                    #### add salt and peppern noise to given images 
    def add_salt_and_pepper_noise(self, image, salt_ratio, pepper_ratio): 
        """
        Adds salt and pepper noise to an image.

        Args:
            image (numpy.ndarray): Input image.
            salt_ratio (float): Ratio of salt noise (default: 0.05).
            pepper_ratio (float): Ratio of pepper noise (default: 0.05).

        Returns:
            numpy.ndarray: Image with salt and pepper noise.
        """
        pepper_ratio=self.noise_intensity
        salt_ratio=self.noise_intensity
        row, col = image.shape
        salt = np.random.rand(row, col) < salt_ratio
        pepper = np.random.rand(row, col) < pepper_ratio
        noisy_image = np.copy(image)
        noisy_image[salt] = 1
        noisy_image[pepper] = 0
        return noisy_image        
    
    def normalize(self, image):
        for i in range(len(image)):
            image[i] = image[i] - np.min(image[i])
            image[i] = image[i]/((np.max(image[i])+1e-5))
        return image   

    def __getitem__(self, index):
        clean_path = self.clean_paths[index]
        if clean_path.endswith('.png'):
            clean = np.array(cv.imread(clean_path, cv.IMREAD_GRAYSCALE),dtype=np.float16)
        elif clean_path.endswith('.pt'):
            clean = torch.load(clean_path)
            cleani = np.zeros_like(clean)
            cleani[80:-80, 45:-45] = clean[80:-80, 45:-45]
            clean = cleani
        
        if self.noise_type == 'gauss':
            """ In that case, we add the gaussian noise in the sinogram, then noise is uncorrelated in sino domain """
            clean /= np.max(clean, axis = (-1,-2))
            clean_sino = np.array((create_noisy_sinograms(np.expand_dims(clean,0), 512, 0)).squeeze(0))
            noisy = np.asarray(add_correlated_noise(clean_sino, self.noise_intensity, self.noise_sigma))
            noisier = np.asarray(add_correlated_noise(noisy, self.noise_intensity, self.noise_sigma))
            noise = clean_sino - noisy
            del(clean_sino)


        elif self.noise_type == 'poisson':
            noisy = np.random.poisson(clean * 1 * self.noise_intensity) / self.noise_intensity / 1.
            # Add Poisson
            noisier = noisy + (np.random.poisson(clean * 1 * self.noise_intensity) / self.noise_intensity / 255. - clean)


        else:
            raise NotImplementedError('wrong type of noise')
        if self.noise_type == "salt_and_pepper":
            clean, noisy, noisier = self.transform(clean), self.transform(noisy), self.transform(noisier)
            clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        elif self.noise_type == "gauss" or self.noise_type == 'gauss_image':
            clean, noisy, noisier = self.transform(clean),  self.transform(noisy), self.transform(noisier)
            clean, noisy, noisier = clean.type(torch.FloatTensor), noisy.type(torch.FloatTensor), noisier.type(torch.FloatTensor)
        else:
            raise NotImplementedError('wrong type of noise')
        if self.noise_type == "salt_and_pepper":
            return {'clean': clean, 'noisy': noisy, 'noisier': noisier}
        elif self.noise_type == "gauss" or self.noise_type == "gauss_image":
            return {'clean': clean, 'noisy': noisy, 'noisier': noisier, 'noise': noise}
            del(clean, noisy, noisier, noise)
        else:
            raise NotImplementedError('wrong type of noise')
    def __len__(self):
        return len(self.clean_paths)



