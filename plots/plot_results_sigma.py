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
from .model import *
from .dataset_EMD import *
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
    default="Heart",
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
    default="/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Test/",
)
parser.add_argument(
    "-w",
    "--weights_dir",
    type=str,
    help="directory to save model weights",
    default="/home/nadja/tomo_project/Results_Noisier2Inverse_Heart/Model_Weights",
)
parser.add_argument(
    "-noise_sigma",
    "--noise_sigma",
    type=float,
    help="how big is the kernel size of convolution",
    default=3.0,
)
parser.add_argument(
    "-noise_sigma_test",
    "--noise_sigma_test",
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

args = parser.parse_args()

# Create output directory if not exists
output_dir = os.path.join(args.outputdir, "Test_Results")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
i = 0

PSNR_sob_z_list = []
PSNR_sob_y_list = []
SSIM_sob_z_list = []
SSIM_sob_y_list = []
PSNR_inf_z_list = []
PSNR_inf_y_list = []
SSIM_inf_z_list = []
SSIM_inf_y_list = []
PSNR_z_list = []
PSNR_y_list = []
SSIM_z_list = []
SSIM_y_list = []
# Initialize lists to store data for plotting
noisy_list, clean_list = [], []
result_sob_y_list, result_sob_z_list = [], []
result_inf_y_list, result_inf_z_list = [], []
result_y_list, result_z_list = [], []
for sigma in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:

    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    path = os.path.join(args.outputdir,"Test_Results_sigmas")

    files = os.listdir(path)
    print(files, flush = True)

    # Loop through files to load the appropriate data
    for f in files:
        if (
            "Sob" in f
            and f.endswith(str(sigma))
            and "_sigma_" + str(args.noise_sigma) in f
        ):
            print('found')       
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_sob_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_sob_y = data["output_reco_array"]

            noisy_list.append(noisy)
            clean_list.append(clean)

            ssim_values_sob_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values

            psnr_values_sob_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))
            ### append mean values psnr to list
            ssim_values_sob_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values

            psnr_values_sob_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

            PSNR_sob_y_list.append(np.mean(psnr_values_sob_y))
            SSIM_sob_y_list.append(np.mean(ssim_values_sob_y))
            PSNR_sob_z_list.append(np.mean(psnr_values_sob_z))
            SSIM_sob_z_list.append(np.mean(ssim_values_sob_z))

        elif (
            "Inference" in f
            and f.endswith(str(sigma))
            and "Sob" not in f
            and "_sigma_" + str(args.noise_sigma) in f
        ):
            print('inf found!', flush = True)
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_inf_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_inf_y = data["output_reco_array"]
            result_inf_y_list.append(result_inf_y)
            ssim_values_inf_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_inf_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

            PSNR_inf_z_list.append(np.mean(psnr_values_inf_z))
            PSNR_inf_y_list.append(np.mean(psnr_values_inf_y))
            SSIM_inf_z_list.append(np.mean(ssim_values_inf_z))
            SSIM_inf_y_list.append(np.mean(ssim_values_inf_y))


        elif (
            f.endswith(str(sigma))
            and "Inf" not in f
            and "Sob" not in f
            and "_sigma_" + str(args.noise_sigma) in f
        ):
            print(f + " we load inference weights for " + str(sigma))
            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_y = data["output_reco_array"]
            result_y_list.append(result_y)
            ssim_values_z = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_z = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_y = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_y = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

            PSNR_z_list.append(np.mean(psnr_values_z))
            PSNR_y_list.append(np.mean(psnr_values_y))
            SSIM_z_list.append(np.mean(ssim_values_z))
            SSIM_y_list.append(np.mean(ssim_values_y))

print("liste " + str(len(SSIM_z_list)), flush=True)
print(len(ssim_values_inf_y), flush = True)

# Set the style
plt.style.use("seaborn-darkgrid")

# Create x-values from 1 to 5
x_values = np.arange(1, 7)


import seaborn as sns

# Use a colorblind-friendly palette
colorblind_palette = sns.color_palette("colorblind", n_colors=3)

plt.figure(figsize=(7, 16))  # Adjust figure size for horizontal layout
plt.style.use("seaborn-darkgrid")

# Set the background color to white
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
marker_style = 'o'  # Example: 'o' for circle, 's' for square, '^' for triangle
marker_style1 = 'x'
# Plot PSNR values
plt.subplot(2, 1, 1)
plt.grid(True, linestyle="--", color="gray", alpha=0.9, axis="y")  # Emphasized horizontal grid lines
plt.axhline(0, color="black", linewidth=1.2, linestyle="-")  # Add a horizontal line for the x-axis
plt.plot(
    x_values,
    PSNR_sob_z_list,
    label="PSNR ours sobo z",
    color=colorblind_palette[0],
    linewidth=2.5,
    marker=marker_style,
    markersize=7,
)
plt.plot(
    x_values,
    PSNR_sob_y_list,
    label="PSNR ours sobo y",
    color=colorblind_palette[0],
    linewidth=2.5,
    linestyle = '--',
    marker=marker_style1,
    markersize=7,
)
plt.plot(
    x_values,
    PSNR_inf_z_list,
    label="PSNR ours z",
    color=colorblind_palette[1],
    linewidth=2.5,
    marker=marker_style,
    markersize=7,
)
plt.plot(
    x_values,
    PSNR_inf_y_list,
    label="PSNR ours y",
    color=colorblind_palette[1],
    linewidth=2.5,
    linestyle = '--',
    marker=marker_style1,
    markersize=7,
)
plt.plot(
    x_values,
    PSNR_z_list,
    label="PSNR z",
    color=colorblind_palette[2],
    linewidth=2.5,
    marker=marker_style,
    markersize=7,
)
plt.plot(
    x_values,
    PSNR_y_list,
    label="PSNR y",
    color=colorblind_palette[2],
    linewidth=2.5,
    linestyle = '--',
    marker=marker_style1,
    markersize=7,
)
plt.title("PSNR values trained on σ " + str(args.noise_sigma), fontsize=32, pad=15)
plt.xlabel("σ", fontsize=22)
plt.ylabel("PSNR (dB)", fontsize=22)
plt.yticks(fontsize=22)  # Adjust y-axis tick label size here
plt.xticks(x_values, fontsize=22)
plt.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.8,
    shadow=True,
    fontsize=20,
)

# Plot SSIM values
plt.subplot(2, 1, 2)
plt.grid(True, linestyle="--", color="gray", alpha=0.9, axis="y")  # Emphasized horizontal grid lines
plt.axhline(0, color="black", linewidth=1.2, linestyle="-")  # Add a horizontal line for the x-axis
plt.plot(
    x_values,
    SSIM_sob_z_list,
    label="SSIM ours sobo z",
    color=colorblind_palette[0],
    linewidth=2.5,
    marker=marker_style,
    markersize=7,
)
plt.plot(
    x_values,
    SSIM_sob_y_list,
    label="SSIM ours sobo y",
    color=colorblind_palette[0],
    linewidth=2.5,
    linestyle = '--',
    marker=marker_style1,
    markersize=7,
)
plt.plot(
    x_values,
    SSIM_inf_z_list,
    label="SSIM ours z",
    color=colorblind_palette[1],
    linewidth=2.5,
    marker=marker_style,
    markersize=7,
)
plt.plot(
    x_values,
    SSIM_inf_y_list,
    label="SSIM ours y",
    color=colorblind_palette[1],
    linewidth=2.5,
    linestyle = '--',
    marker=marker_style1,
    markersize=7,
)
plt.plot(
    x_values,
    SSIM_z_list,
    label="SSIM z",
    color=colorblind_palette[2],
    linewidth=2.5,
    marker=marker_style,
    markersize=7,
)
plt.plot(
    x_values,
    SSIM_y_list,
    label="SSIM y",
    color=colorblind_palette[2],
    linewidth=2.5,
    marker=marker_style1,
    linestyle = '--',
    markersize=7,
)
plt.title("SSIM values trained on σ " + str(args.noise_sigma), fontsize=32, pad=15)
plt.xlabel("σ", fontsize=22)
plt.ylabel("SSIM", fontsize=22)
plt.xticks(x_values, fontsize=22)
plt.yticks(fontsize=22)  # Adjust y-axis tick label size here

plt.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.8,
    shadow=True,
    fontsize=20,
)

# Save the plot
plt.tight_layout(pad=2.0)
plt.savefig(
    os.path.join(
        args.outputdir + "/Plots_Paper",
        "Plot_ssim_psnr_sigmas_" + str(args.noise_sigma) + ".svg",
    ),
    format="svg",
)
