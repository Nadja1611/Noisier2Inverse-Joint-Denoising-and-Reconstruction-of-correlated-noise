# -*- coding: utf-8 -*-
"""
Script for plotting reconstructions and metrics (SSIM, PSNR, EMD)
"""

# %%
import torch
import os
import matplotlib.pyplot as plt
import skimage.metrics as skm
from skimage.data import shepp_logan_phantom
import logging
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from model import *
from dataset_EMD import *
from utils_inverse import create_noisy_sinograms
import argparse
from torch.utils.data import DataLoader
import psutil
from ts_algorithms import fbp, tv_min2d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Parsing arguments for testing
parser = argparse.ArgumentParser(
    description="Arguments for testing denoising network.", add_help=False
)
parser.add_argument(
    "-l",
    "--loss_variant",
    type=str,
    help="which loss variant should be used",
    default="DataDomain_NW_Data_MSE",
)
parser.add_argument(
    "-dataset",
    "--dataset",
    type=str,
    help="which dataset should be used",
    default="Nuts",
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
    help="number of projection angles sinogram",
    default=512,
)
parser.add_argument(
    "-batch_size", "--batch_size", type=int, help="batch size for testing", default=6
)
parser.add_argument(
    "-datadir",
    "--datadir",
    type=str,
    help="directory for data loading",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/",
)
parser.add_argument(
    "-noise_type",
    "--noise_type",
    type=str,
    help="type of noise",
    default="uncorrelated",
)
parser.add_argument(
    "-noise_intensity",
    "--noise_intensity",
    type=float,
    help="intensity of salt and pepper noise",
    default=0.05,
)
parser.add_argument(
    "-out",
    "--outputdir",
    type=str,
    help="directory for saving results",
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_Final/",
)

parser.add_argument(
    "-noise_sigma",
    "--noise_sigma",
    type=float,
    help="how big is the kernel size of convolution",
    default=3.0,
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    help="which learning rate should be used",
    default=1e-5,
)
parser.add_argument(
    "-path",
    "--path",
    type=str,
    help="where do Noisier2Inverse results lie",
    default='/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_final/',
)

parser.add_argument(
    "-path2",
    "--path2",
    type=str,
    help="where do Noise2Inverse results lie",
    default='/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noise2Inverse/Results_Noise2Inverse/',
)


args = parser.parse_args()
# Set up the path where the results are stored
path =  os.path.join(args.path, 'Test_Results')
path2 = os.path.join(args.path2, 'Test_Results')
# Create output directory if not exists
if args.dataset == 'Nuts':
    index = 7
else:
    index = 12    

"""---------------------------- now same for z plots ------------------------------"""
# Create output directory if not exists

i = 0

for sigma in [2, 3, 5]:
    """specify weight directory"""
    print(i, flush = True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(sigma, flush = True)

    files = os.listdir(path)
    files_n2i = os.listdir(path2)

    # Initialize lists to store data for plotting
    noisy_list, clean_list = [], []
    result_sob_y_list, result_sob_z_list = [], []
    result_inf_y_list, result_inf_z_list = [], []
    result_y_list, result_z_list = [], []
    print(files, flush=True)
    # Loop through files to load the appropriate data
    for f in files:
        print(f, flush=True)
        if "Sob" in f and "sigma_" + str(sigma) in f:
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_sob_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_sob_y = data["output_reco_array"]

            ssim_values_sob_z = np.load(os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_sob_y = np.load(os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

        elif "Inference" in f and "sigma_" + str(sigma) in f and "Sob" not in f:
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_inf_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_inf_y = data["output_reco_array"]

            ssim_values_inf_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_inf_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

        elif "sigma_" + str(sigma) in f and "Inf" not in f and "Sob" not in f:
            print(f + " we load inference weights for " + str(sigma))
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_y = data["output_reco_array"]

            ssim_values_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

    ##### load results of noise2inverse
    for f in files_n2i:
        if "sigma_" + str(sigma) in f:
            print("we load n2i", flush=True)
            data = np.load(os.path.join(path2, f, "output_reco_results.npz"))
            result_n2i = data["output_reco_array"]
            ssim_values_n2i = np.load(
                os.path.join(path2, f, "ssim_z.npy")
            )  # Changed ssim to ssim_values
            psnr_values_n2i = np.load(os.path.join(path2, f, "psnr_z.npy"))


    # Create a box plot for PSNR and SSIM values
 
    plt.figure(figsize=(12, 6))

    # Plot PSNR values
    plt.subplot(1, 2, 1)
    plt.plot(psnr_values_sob_z, label="PSNR Sob z " + str(np.mean(psnr_values_sob_z)) + ', ' + str(np.round(np.std(psnr_values_sob_z),3)) )
    plt.plot(psnr_values_sob_y, label="PSNR Sob y" + str(np.mean(psnr_values_sob_y))+ ', ' + str(np.round(np.std(psnr_values_sob_y),3)) )
    plt.plot(psnr_values_inf_z, label="PSNR Inf z" + str(np.mean(psnr_values_inf_z))+ ', ' + str(np.round(np.std(psnr_values_inf_z),3)) )
    plt.plot(psnr_values_inf_y, label="PSNR Inf y" + str(np.mean(psnr_values_inf_y))+ ', ' + str(np.round(np.std(psnr_values_inf_y),3)) )
    plt.plot(psnr_values_z, label="PSNR z" + str(np.mean(psnr_values_z))+ ', ' + str(np.round(np.std(psnr_values_z),3)) )
    plt.plot(psnr_values_y, label="PSNR y" + str(np.mean(psnr_values_y))+ ', ' + str(np.round(np.std(psnr_values_y),3)) )
    plt.plot(psnr_values_n2i, label="PSNR N2I" + str(np.mean(psnr_values_n2i))+ ', ' + str(np.round(np.std(psnr_values_n2i),3)) )
    plt.title("PSNR Values sigma " + str(sigma))
    plt.xlabel("Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()

    # Plot SSIM values
    plt.subplot(1, 2, 2)
    plt.plot(ssim_values_sob_z, label="SSIM Sob z" + str(np.mean(ssim_values_sob_z))+ ', ' + str(np.round(np.std(ssim_values_sob_z),3)) )
    plt.plot(ssim_values_sob_y, label="SSIM Sob y" + str(np.mean(ssim_values_sob_y))+ ', ' + str(np.round(np.std(ssim_values_sob_y),3)) )
    plt.plot(ssim_values_inf_z, label="SSIM Inf z" + str(np.mean(ssim_values_inf_z))+ ', ' + str(np.round(np.std(ssim_values_inf_z),3)) )
    plt.plot(ssim_values_inf_y, label="SSIM Inf y" + str(np.mean(ssim_values_inf_y))+ ', ' + str(np.round(np.std(psnr_values_inf_y),3)) )
    plt.plot(ssim_values_z, label="SSIM z" + str(np.mean(ssim_values_z))+ ', ' + str(np.round(np.std(ssim_values_z),3)) )
    plt.plot(ssim_values_y, label="SSIM y" + str(np.mean(ssim_values_y))+ ', ' + str(np.round(np.std(ssim_values_y),3)) )
    plt.plot(ssim_values_n2i, label="SSIM N2I" + str(np.mean(ssim_values_n2i))+ ', ' + str(np.round(np.std(ssim_values_n2i),3)) )
    plt.title("SSIM Values sigma " + str(sigma))
    plt.xlabel("Index")
    plt.ylabel("SSIM")
    plt.legend()
    vmin = 0
    vmax =1

    plt.savefig(
        os.path.join(
            args.outputdir + "/Plots_Paper",
            "Plot_ssim_psnr_last" + str(sigma) + ".png",
        )
    )


    plt.close()


    titles_y = [
        "noisy ($\\sigma$ = " + str(sigma) + ")",
        "clean",
        "ours sobo y",
        "ours y",
        "y",
        "N2I",
    ]

    # Create a figure and axes with specified size and no spacing
    if i == 0:

        fig, axes = plt.subplots(3, 6, figsize=(10, 5), constrained_layout=True)
        plt.subplots_adjust(wspace=0, hspace=0)  # Adjust spacing

    for j in range(6):  # 6 columns
        # Determine the data for each subplot
        if j == 0 and i > 0:
            data = noisy[index][10:-10, 10:-10]
        elif j == 0 and i == 0:
            data = noisy[index][10:-10, 10:-10]
        elif j == 1:
            data = clean[index][10:-10, 10:-10]
        elif j == 2:
            data = result_sob_y[index][10:-10, 10:-10]
        elif j == 3:
            data = result_inf_y[index][10:-10, 10:-10]
        elif j == 4:
            data = result_y[index][10:-10, 10:-10]
        elif j == 5:
            data = result_n2i[index][10:-10, 10:-10]

        # Display the main image
        if j != 0:
            axes[i, j].imshow(data, cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
        else:
            axes[i, j].imshow(data, cmap="gray")
            axes[i, j].axis("off")
        # Set title only for the first row
        if i == 0:
            axes[i, j].set_title(titles_y[j], fontsize=12)

        # Create the inset axes for each subplot (for zoomed-in view)
        axins = inset_axes(
            axes[i, j],
            width=0.6,   # Width of the inset axes (adjust as needed)
            height=0.6,  # Height of the inset axes (adjust as needed)
            loc="upper right",  # Location of the inset within the subplot
            borderpad=-0.5  # Adjust the distance between the main plot and the inset
        )
        if 'Heart' not in args.outputdir:
        # Define zoom region (adjust x1, x2, y1, y2 as needed for each image)
            x1, x2, y1, y2 = 160, 210, 160, 210  # Zoom region coordinates (example)
        else:
            x1, x2, y1, y2 = 140, 190, 120, 170  # Zoom region coordinates (example)

        # Display the zoomed-in region
        axins.imshow(data, cmap="gray", vmin=0, vmax = 1)
        axins.set_xlim(x1, x2)  # Set x-limits for the zoomed region
        axins.set_ylim(y2, y1)  # Set y-limits for the zoomed region

        # Remove ticks from the inset for a cleaner look
        axins.set_xticks([])
        axins.set_yticks([])

        # Indicate the zoomed region on the main axes (with a red rectangle or other indicator)
        axes[i, j].indicate_inset_zoom(axins, edgecolor="red")  # Red edge for zoom box

    # Set the background color for the figure
    fig.patch.set_facecolor("white")

    # Adjust layout for better spacing and save the figure
 #   plt.tight_layout()


    plt.savefig(
        os.path.join(
            args.outputdir + "/Plots_Paper",
            "last_Plot_all" + "_y.svg",
        )
    )
    plt.savefig(
        os.path.join(
            args.outputdir + "/Plots_Paper",
            "last_Plot_all" + "_y.png",
        )
    )
    i +=1










"""---------------------------- now same for z plots ------------------------------"""
# Create output directory if not exists
i = 0

for sigma in [2, 3, 5]:
    """specify weight directory"""
    print(i, flush = True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(sigma, flush = True)

    files = os.listdir(path)
    files_n2i = os.listdir(path2)

    # Initialize lists to store data for plotting
    noisy_list, clean_list = [], []
    result_sob_y_list, result_sob_z_list = [], []
    result_inf_y_list, result_inf_z_list = [], []
    result_y_list, result_z_list = [], []
    print(files, flush=True)
    # Loop through files to load the appropriate data
    for f in files:
        print(f, flush=True)
        if "Sob" in f and "sigma_" + str(sigma) in f:
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_sob_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_sob_y = data["output_reco_array"]

            ssim_values_sob_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_sob_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

        elif "Inference" in f and "sigma_" + str(sigma) in f and "Sob" not in f:
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_inf_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_inf_y = data["output_reco_array"]

            ssim_values_inf_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_inf_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

        elif "sigma_" + str(sigma) in f and "Inf" not in f and "Sob" not in f:
            print(f + " we load inference weights for " + str(sigma))
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_y = data["output_reco_array"]

            ssim_values_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

    ##### load results of noise2inverse
    for f in files_n2i:
        if "sigma_" + str(sigma) in f:
            print("we load n2i", flush=True)
            data = np.load(os.path.join(path2, f, "output_reco_results.npz"))
            result_n2i = data["output_reco_array"]
            ssim_values_n2i = np.load(
                os.path.join(path2, f, "ssim_z.npy")
            )  # Changed ssim to ssim_values
            psnr_values_n2i = np.load(os.path.join(path2, f, "psnr_z.npy"))


    # Create a box plot for PSNR and SSIM values
 
    plt.figure(figsize=(12, 6))

    # Plot PSNR values
    plt.subplot(1, 2, 1)
    plt.plot(psnr_values_sob_z, label="PSNR Sob z " + str(np.mean(psnr_values_sob_z)) + ', ' + str(np.round(np.std(psnr_values_sob_z),3)) )
    plt.plot(psnr_values_sob_y, label="PSNR Sob y" + str(np.mean(psnr_values_sob_y))+ ', ' + str(np.round(np.std(psnr_values_sob_y),3)) )
    plt.plot(psnr_values_inf_z, label="PSNR Inf z" + str(np.mean(psnr_values_inf_z))+ ', ' + str(np.round(np.std(psnr_values_inf_z),3)) )
    plt.plot(psnr_values_inf_y, label="PSNR Inf y" + str(np.mean(psnr_values_inf_y))+ ', ' + str(np.round(np.std(psnr_values_inf_y),3)) )
    plt.plot(psnr_values_z, label="PSNR z" + str(np.mean(psnr_values_z))+ ', ' + str(np.round(np.std(psnr_values_z),3)) )
    plt.plot(psnr_values_y, label="PSNR y" + str(np.mean(psnr_values_y))+ ', ' + str(np.round(np.std(psnr_values_y),3)) )
    plt.plot(psnr_values_n2i, label="PSNR N2I" + str(np.mean(psnr_values_n2i))+ ', ' + str(np.round(np.std(psnr_values_n2i),3)) )
    plt.title("PSNR Values sigma " + str(sigma))
    plt.xlabel("Index")
    plt.ylabel("PSNR (dB)")
    plt.legend()

    # Plot SSIM values
    plt.subplot(1, 2, 2)
    plt.plot(ssim_values_sob_z, label="SSIM Sob z" + str(np.mean(ssim_values_sob_z))+ ', ' + str(np.round(np.std(ssim_values_sob_z),3)) )
    plt.plot(ssim_values_sob_y, label="SSIM Sob y" + str(np.mean(ssim_values_sob_y))+ ', ' + str(np.round(np.std(ssim_values_sob_y),3)) )
    plt.plot(ssim_values_inf_z, label="SSIM Inf z" + str(np.mean(ssim_values_inf_z))+ ', ' + str(np.round(np.std(ssim_values_inf_z),3)) )
    plt.plot(ssim_values_inf_y, label="SSIM Inf y" + str(np.mean(ssim_values_inf_y))+ ', ' + str(np.round(np.std(psnr_values_inf_y),3)) )
    plt.plot(ssim_values_z, label="SSIM z" + str(np.mean(ssim_values_z))+ ', ' + str(np.round(np.std(ssim_values_z),3)) )
    plt.plot(ssim_values_y, label="SSIM y" + str(np.mean(ssim_values_y))+ ', ' + str(np.round(np.std(ssim_values_y),3)) )
    plt.plot(ssim_values_n2i, label="SSIM N2I" + str(np.mean(ssim_values_n2i))+ ', ' + str(np.round(np.std(ssim_values_n2i),3)) )
    plt.title("SSIM Values sigma " + str(sigma))
    plt.xlabel("Index")
    plt.ylabel("SSIM")
    plt.legend()
    vmin = 0
    vmax =1



    plt.close()
    titles_z = [
        "noisy ($\\sigma$ = " + str(sigma) + ")",
        "clean",
        "ours sobo z",
        "ours z",
        "z",
        "N2I",
    ]

    # Create a figure and axes with specified size and no spacing
    if i == 0:

        fig, axes = plt.subplots(3, 6, figsize=(10, 5), constrained_layout=True)
        plt.subplots_adjust(wspace=0, hspace=0)  # Adjust spacing

    for j in range(6):  # 6 columns
        # Determine the data for each subplot
        if j == 0 and i > 0:
            data = noisy[index][10:-10, 10:-10]
        elif j == 0 and i == 0:
            data = noisy[index][10:-10, 10:-10]
        elif j == 1:
            data = clean[index][10:-10, 10:-10]
        elif j == 2:
            data = result_sob_z[index][10:-10, 10:-10]
        elif j == 3:
            data = result_inf_z[index][10:-10, 10:-10]
        elif j == 4:
            data = result_z[index][10:-10, 10:-10]
        elif j == 5:
            data = result_n2i[index][10:-10, 10:-10]

        # Display the main image
        if j != 0:
            axes[i, j].imshow(data, cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
        else:
            axes[i, j].imshow(data, cmap="gray")
            axes[i, j].axis("off")
        # Set title only for the first row
        if i == 0:
            axes[i, j].set_title(titles_z[j], fontsize=12)

        # Create the inset axes for each subplot (for zoomed-in view)
        axins = inset_axes(
            axes[i, j],
            width=0.6,   # Width of the inset axes (adjust as needed)
            height=0.6,  # Height of the inset axes (adjust as needed)
            loc="upper right",  # Location of the inset within the subplot
            borderpad=-0.5  # Adjust the distance between the main plot and the inset
        )
        if 'Heart' not in args.outputdir:
        # Define zoom region (adjust x1, x2, y1, y2 as needed for each image)
            x1, x2, y1, y2 = 160, 210, 160, 210  # Zoom region coordinates (example)
        else:
            x1, x2, y1, y2 = 140, 190, 120, 170  # Zoom region coordinates (example)

        # Display the zoomed-in region
        axins.imshow(data, cmap="gray", vmin=0, vmax = 1)
        axins.set_xlim(x1, x2)  # Set x-limits for the zoomed region
        axins.set_ylim(y2, y1)  # Set y-limits for the zoomed region

        # Remove ticks from the inset for a cleaner look
        axins.set_xticks([])
        axins.set_yticks([])

        # Indicate the zoomed region on the main axes (with a red rectangle or other indicator)
        axes[i, j].indicate_inset_zoom(axins, edgecolor="red")  # Red edge for zoom box

    # Set the background color for the figure
    fig.patch.set_facecolor("white")

    # Adjust layout for better spacing and save the figure
 #   plt.tight_layout()


    plt.savefig(
        os.path.join(
            args.outputdir + "/Plots_Paper",
            "Last_Plot_all" + "_z.svg",
        )
    )
    plt.savefig(
        os.path.join(
            args.outputdir + "/Plots_Paper",
            "Last_Plot_all" + "_z.png",
        )
    )
    i +=1


    
    
    
    
