# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:51:34 2024

"""

# %%
import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import skimage.metrics as skm
from skimage.data import shepp_logan_phantom
import logging
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from model import *
from torch.optim import lr_scheduler
from itertools import combinations
import LION.CTtools.ct_utils as ct
from ts_algorithms import fbp, tv_min2d
import skimage
import argparse
import gc
from scipy.ndimage import gaussian_filter
from dataset_EMD import *
from utils_inverse import create_noisy_sinograms
from matplotlib.ticker import MaxNLocator
import psutil



parser = argparse.ArgumentParser(
    description="Arguments for denoising network.", add_help=False
)
parser.add_argument(
    "-l",
    "--loss_variant",
    type=str,
    help="which loss variant should be used",
    default="DataDomain_NW_Data_MSE",
)
parser.add_argument(
    "-alpha",
    "--alpha",
    type=float,
    help="how much noisier should z be than y",
    default=1,
)
parser.add_argument(
    "-angles",
    "--angles",
    type=int,
    help="number of prosqueuejection angles sinogram",
    default=128,
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    type=int,
    help="number of prosqueuejection angles sinogram",
    default=6,
)
parser.add_argument(
    "-datadir",
    "--datadir",
    type=str,
    help="from where should the data be loaded",
    default='/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/',
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-5,
)
parser.add_argument(
    "-noise_type",
    "--noise_type",
    type=str,
    help="add correlated or uncorrelated noise",
    default="uncorrelated",
)
parser.add_argument(
    "-noise_intensity",
    "--noise_intensity",
    type=float,
    help="how intense should salt and pepper noise be",
    default=0.05,
)
parser.add_argument(
    "-noise_sigma",
    "--noise_sigma",
    type=float,
    help="what sigma has filtering",
    default=0.05,
)
parser.add_argument(
    "-o",
    "--logdir",
    type=str,
    help="directory for log files",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs",
)
parser.add_argument(
    "-w",
    "--weights_dir",
    type=str,
    help="directory to save model weights",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2eInverse/Model_Weights",
)

parser.add_argument(
    "-out",
    "--outputdir",
    type=str,
    help="directory where results are saved",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper/",
)

args = parser.parse_args()

""" explanation loss_variant"""
## DataDomain_MSE         - loss is computed in data domain, NW operates in recon domain
## ReconDomain_MSE        - loss is computed in recon domain, NW operates in recon domain

## DataDomain_MSE_Inference           - loss is computed in data domain, NW operates in recon domain, we use inference loss
## ReconDomain_MSE_Inference          - loss is computed in recon domain, NW operates in recon domain, we use inference loss
## DataDomain_MSE_Inference_Sobolev           - Sobolev loss is computed in data domain, NW operates in recon domain, we use inference loss

"""specify weight directory"""
weights_dir = (
    args.weights_dir
    + "/Noise_"
   + args.noise_type
    + "_sigma_"
    + str(args.noise_sigma)
    + '_' 
    + str(args.noise_intensity)
    + "_batchsize_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_alpha_"
    + str(args.alpha)
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

device = "cuda:0"

print(args.datadir, flush = True)


class Noiser2NoiseRecon:
    def __init__(
        self,
        device: str = "cuda:0",
        folds: int = 1,
    ):
        self.net_denoising = UNet(in_channels=1, out_channels=1).to(device)
        self.folds = folds
        self.device = device
        self.angles = args.angles
        self.batch_size = args.batch_size

        # speicift noise type and intensity
        self.noise = args.noise_type
        self.noise_intensity = args.noise_intensity

        # Dataset
        self.train_dataset = Walnut(
            noise_type=self.noise, noise_intensity=self.noise_intensity, noise_sigma = args.noise_sigma, train=True, data_dir = args.datadir
        )
        print(self.train_dataset.clean_paths,flush = True)
        self.test_dataset = Walnut(
            noise_type=self.noise, noise_intensity=self.noise_intensity, noise_sigma = args.noise_sigma, train=False, data_dir = args.datadir
        )


        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers = 0,
        )

    def forward(self, reconstruction):
        #### the input could be the image or the sinogram, it is just called reconstruction as for most cases it is the reconstructionß
        output_denoising = self.net_denoising(reconstruction.float().to(self.device))
        # if we use the method where network operates in reconstruction domain, we have to project images back again
        output_denoising_reco = output_denoising
        output_denoising_sino = self.projection_tomosipo(
            output_denoising, sino=self.angles
        )  # .to(self.device)

        return output_denoising_reco, output_denoising_sino

    def compute_reconstructions(self, sinograms):
        sinograms = sinograms.squeeze(1)  # .detach().cpu()

        Reconstructions = torch.zeros(
            (sinograms.shape[0], self.folds, sinograms.shape[-2], sinograms.shape[-2]),
            device=sinograms.device,
        )
        number_of_angles = sinograms.shape[-1]
        projection_indices = np.array([i for i in range(0, number_of_angles)])
        for i in range(sinograms.shape[0]):
            Reconstructions[i] = self.fbp_tomosipo(
                sinograms[i].unsqueeze(0).unsqueeze(0),
                translate=False,
                angle_vector=projection_indices,
                folds=1,
            )
        del sinograms
        return Reconstructions

    def projection_tomosipo(self, img, sino, translate=False):
        if isinstance(sino, int) == True:
            angles = sino
        else:
            angles = sino.shape[-1]
        # 0.1: Make geometry:
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(img.shape[0], img.shape[-1], img.shape[-1]),
            number_of_angles=angles,
            translate=False,
        )
        # 0.2: create operator:
        op = to_autograd(ct.make_operator(geo))
        sino = op((img[:, 0]).to(self.device))
        sino = sino.unsqueeze(1)
        sino = torch.moveaxis(sino, -1, -2)
        return sino

    def fbp_tomosipo(self, sino, angle_vector=None, translate=False, folds=None):
        angles = sino.shape[-1]
        geo = ctgeo.Geometry.parallel_default_parameters(
            image_shape=(sino.shape[0], 336, 336),
            number_of_angles=angles,
            translate=False,
            angle_vector=angle_vector,
        )
        op = ct.make_operator(geo)
        sino = torch.moveaxis(sino, -1, -2)

        result = fbp(op, sino[:, 0])
        result = result.unsqueeze(1)
        del (sino, op, angles, geo)

        return result


#### generate a new output path where the results are stored!
newpath = (args.outputdir + "/Noise_"
    + args.noise_type
    + "_sigma_"
    + str(args.noise_sigma)
    + '_' 
    + str(args.noise_intensity)
    + "_batchsize_"
    + str(args.batch_size)
    + "_"
    + "Gaussian_Method_"
    + args.loss_variant
    + "_alpha_"
    + str(args.alpha)
    + "_learning_rate_"
    + str(args.learning_rate)
    + "_angles_"
    + str(args.angles)
)
if not os.path.exists(newpath):
    os.makedirs(newpath)


###### specifiy training parameters
N_epochs = 10000
learning_rate = args.learning_rate


###### Choose from 'MSE_image', 'MSE_data', 'Sobolev_data'
N2NR = Noiser2NoiseRecon()
N2NR_optimizer = optim.Adam(N2NR.net_denoising.parameters(), lr=learning_rate)


########################### Now training starts ##############
l2_list = []
all_MSEs = []
# Initialize empty tensors for accumulating mean values
all_ssim_y = torch.tensor([])
all_ssim_z = torch.tensor([])

all_psnr_y = torch.tensor([])
all_psnr_z = torch.tensor([])

all_emd_z = torch.tensor([])
all_emd_y = torch.tensor([])

old_psnr = 0.1
old_ssim = 0.1
old_psnr_y = 0.1
old_ssim_y = 0.1
old_emd_y = 10000
old_emd_z = 10000


for epoch in range(N_epochs):
    print('We are at epoch: ' + str(epoch), flush = True)
    running_loss = 0
    running_L2_loss = 0
    torch.cuda.empty_cache()

    # (N2NR.train_dataloader, desc='Epoch {}'.format(epoch)) as tepoch:
    for batch, data in enumerate(N2NR.train_dataloader):
        N2NR.net_denoising.train()
        clean, y, z = (
            data["clean"].squeeze(),
            data["noisy"].squeeze(),
            data["noisier"].squeeze(),
        )

        if len(y.shape) < 3:
            y = y.unsqueeze(0)
            y = y.unsqueeze(1)
            clean = clean.unsqueeze(0)

        if len(z.shape) < 3:
            z = z.unsqueeze(0)
            z = z.unsqueeze(1)

        else:
            y = y.unsqueeze(1)
            z = z.unsqueeze(1)                      



        # generate recos from noisier data, z in the paper is the noisier sinogram
        z_reco = N2NR.compute_reconstructions(z.to(device))
        # generate recos from noisy given data y
        y_reco = N2NR.compute_reconstructions(y.to(device))
        # put reconstructions onto device
        # z_reco = z_recos.to(device)
        # y_reco = y_recos.to(device)
        y = y.to(device)
        z = z.to(device)

        if epoch % 2000 == 0:
            with torch.no_grad():
                plt.subplot(1, 2, 1)
                plt.imshow(z_reco[0][0].detach().cpu(), cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(y_reco[0][0].detach().cpu(), cmap="gray")
                plt.savefig(newpath + "/" + "recon" + ".png")
                plt.close()
        N2NR_optimizer.zero_grad()
        output_reco, output_sino = N2NR.forward(z_reco)
        if epoch != 10 and epoch % 2000 != 0:
            del output_reco
        if args.loss_variant == "DataDomain_MSE_Inference_EMD_Sobolev":
            loss = sobolev_norm(output_sino.float(),  y.float().detach())

        else:
            loss = torch.nn.functional.mse_loss(
                output_sino.float(),  y.float().detach()
            )

        # compute gradient
        loss.backward()
        N2NR_optimizer.step()
        with torch.no_grad():
            running_loss += loss.item()
    torch.cuda.empty_cache()




    if epoch % 2000 == 0:
        with torch.no_grad():
            plt.figure(figsize=(10, 10))
            plt.subplot(221)
            plt.imshow(z_reco[0, 0].detach().cpu())
            plt.colorbar()
            plt.title("z_reco")
            plt.subplot(222)
            plt.imshow(output_reco[0, 0].detach().cpu())
            plt.colorbar()
            plt.title("denoised zreco")
            plt.subplot(223)
            plt.imshow(clean[0].detach().cpu(), aspect="auto")
            plt.colorbar()
            plt.title("clean")
            plt.subplot(224)
            plt.imshow(
                torch.abs(output_sino[0, 0].detach().cpu() - y[0, 0].detach().cpu()),
                aspect="auto",
            )
            plt.savefig(newpath + "/image_" + str(epoch))
            plt.close()
    del (output_sino, clean, z_reco)
    del (z, y, y_reco)

    if epoch % 6 == 0:
        MSEs = []
        ssim_y = []
        ssim_z = []
        ssim_cor = []
        ssim_y_cor = []
        psnr_y = []
        psnr_z = []
        psnr_cor = []
        psnr_y_cor = []
        emd_z = []
        emd_y = []

        with torch.no_grad():
            N2NR.net_denoising.eval()

            for batch, data in tqdm(enumerate(N2NR.test_dataloader)):
                #### apply gassian noise a second time to make the sinogram even noisier
                if args.noise_type == "salt_and_pepper":
                    clean_test, noisy_test, noisier_test = (
                        data["clean"],
                        data["noisy"],
                        data["noisier"],
                    )
                    y_test = torch.tensor(
                        create_noisy_sinograms(
                            np.array(noisy_test.squeeze()), N2NR.angles, 0
                        )
                    )
                    z_test = torch.tensor(
                        create_noisy_sinograms(
                            np.array(noisier_test.squeeze()), N2NR.angles, 0
                        )
                    )
                    clean_test, noisy_test, noisier_test = (
                        clean_test.squeeze(),
                        noisy_test.to(N2NR.device),
                        noisier_test.to(N2NR.device),
                    )
                else:
                    clean_test, y_test, z_test, noise_test = (
                        data["clean"].squeeze(),
                        data["noisy"].squeeze(),
                        data["noisier"].squeeze(),
                        data["noise"].squeeze(),
                    )
                    
                #y_test = y_test.unsqueeze(1)
                # y_test = torch.moveaxis(y_test, -1, -2)
                #z_test = z_test.unsqueeze(1)
                # z_test = torch.moveaxis(z_test, -1, -2)
                if len(y_test.shape) < 3:
                    y_test = y_test.unsqueeze(0)
                    y_test = y_test.unsqueeze(1)
                    clean_test = clean_test.unsqueeze(0)

                if len(z_test.shape) < 3:
                    z_test = z_test.unsqueeze(0)
                    z_test = z_test.unsqueeze(1)


                else:
                    y_test = y_test.unsqueeze(1)
                    z_test = z_test.unsqueeze(1)

                z_test = z_test.to(device)
                y_test = y_test.to(device)
                z_recos_test = N2NR.compute_reconstructions(z_test).detach()
                y_recos_test = N2NR.compute_reconstructions(y_test).detach()

                """corresponds to the case where the neural network operates on recon domain"""
                output_reco, output_sino = N2NR.forward(z_recos_test.to(N2NR.device))
                output_reco = 2*output_reco - z_recos_test.to(output_reco.device)
                output_sino = 2*output_sino - z_test.to(output_sino.device)
                output_reco_y, output_sino_y = N2NR.forward(y_recos_test.to(N2NR.device))

                for i in range(len(clean_test)):
                    # Ensure the tensors are on CPU and converted to numpy arrays
                    ims_test_np = clean_test[i].detach().cpu().numpy()
                    output_reco_y_np = output_reco_y[i][0].detach().cpu().numpy()
                    # we also have a look at the output obtained by directly applying NW to z
                    output_reco_z_np = output_reco[i][0].detach().cpu().numpy()

                # compute the difference between noisy on given data and predicted image (this should follow quite the same noise distribution as the given noise in y)
                    n = output_sino[i] - y_test[i].to(output_sino[i].device)
                    n_y = output_sino_y[i] - y_test[i].to(output_sino_y[i].device)
                    ### We interpret in the case when @inference method is used, the output as corrected one, as correction is in the loss function
                    '''now we compute earth mover distance '''
                    # Convert images to NumPy arrays
                    noise_test_flattened = np.array(noise_test[i]).flatten()  # Flatten the image to 1D
                    n_flattened = np.array(n.detach().cpu()).flatten()
                    n_y_flattened = np.array(n.detach().cpu()).flatten()


                    # Compute Wasserstein distance between the two distributions (images)
                    emd_z_value = wasserstein_distance(noise_test_flattened, n_flattened)
                    emd_y_value = wasserstein_distance(noise_test_flattened, n_y_flattened)


                    # Calculate the data range for SSIM and PSNR
                    data_range = ims_test_np.max() - ims_test_np.min()

                    # Compute SSIM for y and z
                    ssim_y_value = torch.tensor(
                        skimage.metrics.structural_similarity(
                            output_reco_y_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)
                    ssim_z_value = torch.tensor(
                        skimage.metrics.structural_similarity(
                            output_reco_z_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)

                    # Compute PSNR for y and z
                    psnr_y_value = torch.tensor(
                        skimage.metrics.peak_signal_noise_ratio(
                            output_reco_y_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)
                    psnr_z_value = torch.tensor(
                        skimage.metrics.peak_signal_noise_ratio(
                            output_reco_z_np, ims_test_np, data_range=data_range
                        )
                    ).to(device)

                    # Append the computed values to the respective lists
                    ssim_y.append(ssim_y_value)
                    ssim_z.append(ssim_z_value)
                    psnr_y.append(psnr_y_value)
                    psnr_z.append(psnr_z_value)
                    emd_y.append(emd_y_value)
                    
                    emd_z.append(emd_z_value)
                    print('liste')
                    print(emd_z, flush = True)
            # append emd to emd list
            # s
            print(torch.mean(torch.tensor(emd_z)), flush = True)
            all_emd_y = torch.cat(
                                (all_emd_y, torch.tensor([torch.mean(torch.tensor(emd_y))]))
            )
            all_emd_z = torch.cat(
                                 (all_emd_z, torch.tensor([torch.mean(torch.tensor(emd_z))]))
            )

            # Calculate mean values and append to the tensors
            all_ssim_y = torch.cat(
                (all_ssim_y, torch.tensor([torch.mean(torch.tensor(ssim_y))]))
            )
            all_ssim_z = torch.cat(
                (all_ssim_z, torch.tensor([torch.mean(torch.tensor(ssim_z))]))
            )

            all_psnr_y = torch.cat(
                (all_psnr_y, torch.tensor([torch.mean(torch.tensor(psnr_y))]))
            )
            all_psnr_z = torch.cat(
                (all_psnr_z, torch.tensor([torch.mean(torch.tensor(psnr_z))]))
            )
            ### get back
            del (ssim_y, ssim_z, psnr_y, psnr_z, emd_y, emd_z)

            print(psutil.cpu_percent(), flush=True)

            if epoch % 200 == 0:
                with torch.no_grad():
                    plt.figure(figsize=(10, 10))
                    plt.subplot(221)
                    plt.imshow(z_recos_test[0, 0].detach().cpu())
                    plt.colorbar()
                    plt.title("z_reco")
                    plt.subplot(222)
                    plt.imshow(output_reco[0, 0].detach().cpu())
                    plt.colorbar()
                    plt.title("denoised corr")
                    plt.subplot(223)
                    plt.imshow(clean_test[0].detach().cpu())
                    plt.colorbar()
                    plt.title("clean")
                    plt.subplot(224)
                    plt.imshow(output_reco_y[0, 0].detach().cpu(), cmap="gray")
                    plt.colorbar()
                    plt.title("denoised y_reco")
                    plt.show()
                    #plt.savefig(newpath + "/image_val_" + str(epoch))
                    #plt.close()
            if epoch % 40 == 0:
                plt.imshow(output_reco[0, 0].detach().cpu(), cmap="gray")
                plt.show()
                #plt.savefig(newpath + "/results" + str(epoch))
                #plt.close()

            if epoch % 40 == 0:
                plt.imshow(output_reco_y[0, 0].detach().cpu(), cmap="gray")
                plt.show()
                #plt.savefig(newpath + "/results_y" + str(epoch))
                #plt.close()
            """ save model weights if epoch > 200 """
            if epoch > 200:
                if all_ssim_z[-1] > old_ssim:
                    old_ssim = all_ssim_z[-1]
                    weights_path = os.path.join(weights_dir, f"ssim_model_weights.pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights ssim saved at epoch {epoch} to {weights_path}"
                    )
                if all_psnr_z[-1] > old_psnr:
                    old_psnr = all_psnr_z[-1]
                    weights_path = os.path.join(weights_dir, f"psnr_model_weights.pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights psnr saved at epoch {epoch} to {weights_path}"
                    )
                if all_ssim_y[-1] > old_ssim_y:
                    old_ssim_y = all_ssim_y[-1]
                    weights_path = os.path.join(weights_dir, f"ssim_model_weights_y.pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights ssim saved at epoch {epoch} to {weights_path}"
                    )
                if all_psnr_y[-1] > old_psnr_y:
                    old_psnr_y = all_psnr_y[-1]
                    weights_path = os.path.join(weights_dir, f"psnr_model_weights_y.pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights psnr saved at epoch {epoch} to {weights_path}"
                    )
                #### save weights with lowest emd    
                if all_emd_y[-1] < old_emd_y:
                    old_emd_y = all_emd_y[-1]
                    weights_path = os.path.join(weights_dir, f"emd_model_weights_y.pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights emd saved at epoch {epoch} to {weights_path}"
                    )
                if all_emd_z[-1] < old_emd_z:
                    old_emd_z = all_emd_z[-1]
                    weights_path = os.path.join(weights_dir, f"emd_model_weights_z.pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path)
                    print(
                        f"Model weights emd saved at epoch {epoch} to {weights_path}"
                    )
                    
                
                ##### we save weights after each epoch , to get last weights too
                weights_path = os.path.join(weights_dir, f"last.pth")
                torch.save(N2NR.net_denoising.state_dict(), weights_path)
                print(
                    f"Model weights emd saved at epoch {epoch} to {weights_path}"
                )  

                if epoch % 1000 == 0:
                    weights_path = os.path.join(weights_dir, f"weights_" + str(epoch) + ".pth")
                    torch.save(N2NR.net_denoising.state_dict(), weights_path) 

                
                
            #### visualize ssim and psnrs on validation data
            if epoch % 200 == 0 and epoch > 300:
                # Define a larger figure size for better visualization
                plt.figure(figsize=(15, 5))

                # Customize fonts and colors for aesthetics
                plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

                # First subplot: SSIM Validation
                plt.subplot(131)
                plt.plot(all_ssim_z.detach().cpu(), label="Correction (Pred on Z)", color='darkblue', linewidth=2)
                plt.plot(all_ssim_y.detach().cpu(), label="Prediction on Y", color='darkgreen', linestyle='--', linewidth=2)
                plt.legend(loc='best')
                plt.title(f"SSIM Validation", fontsize=14)
                plt.xlabel("Epochs")
                plt.ylabel("SSIM")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

                # Second subplot: PSNR Validation
                plt.subplot(132)
                plt.plot(all_psnr_z.detach().cpu(), label="Prediction on Z", color='darkred', linewidth=2)
                plt.plot(all_psnr_y.detach().cpu(), label="Prediction on Y", color='orange', linestyle='--', linewidth=2)
                plt.legend(loc='best')
                plt.title(f"PSNR Validation ", fontsize=14)
                plt.xlabel("Epochs")
                plt.ylabel("PSNR (dB)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

                # Third subplot: EMD Validation
                plt.subplot(133)
                plt.plot(all_emd_z[300:].detach().cpu(), label="Prediction on Z", color='purple', linewidth=2)
                plt.plot(all_emd_y[300:].detach().cpu(), label="Prediction on Y", color='darkcyan', linestyle='--', linewidth=2)
                plt.legend(loc='best')
                plt.title(f"EMD Validation", fontsize=14)
                plt.xlabel("Epochs")
                plt.ylabel("EMD")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

                # Add an overall title and save the figure
                plt.suptitle(f"SSIM, PSNR, and EMD Validation", fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(newpath + "/ssim_psnr_emd_epoch" + str(epoch))
                plt.close()
            del (ims_test_np, output_reco_y_np, output_reco_z_np)
            torch.cuda.empty_cache()