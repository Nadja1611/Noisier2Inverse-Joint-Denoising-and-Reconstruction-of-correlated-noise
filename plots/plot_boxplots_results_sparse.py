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
            data = np.load(os.path.join(path, f, "output_reco_results_zpsnr_model_weights.pth.npz"))
            result_sob_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ypsnr_model_weights.pth.npz"))
            result_sob_y = data["output_reco_array"]

            ssim_values_sob_z = np.load(
                os.path.join(path, f, "ssim_zpsnr_model_weights.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_z = np.load(os.path.join(path, f, "psnr_zpsnr_model_weights.pth.npy"))

            ssim_values_sob_y = np.load(
                os.path.join(path, f, "ssim_ypsnr_model_weights.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_y = np.load(os.path.join(path, f, "psnr_ypsnr_model_weights.pth.npy"))


            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_sob_z_last = data["output_reco_array"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_sob_y_last = data["output_reco_array"]

            ssim_values_sob_z_last = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_z_last = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_sob_y_last = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_sob_y_last = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))
            psnr_values_input =  np.load(os.path.join(path, f, "input_psnr_.npy"))
            ssim_values_input =  np.load(os.path.join(path, f, "input_ssim_.npy"))


        elif "Inference" in f and "sigma_" + str(sigma) in f and "Sob" not in f:
            data = np.load(os.path.join(path, f, "output_reco_results_zpsnr_model_weights.pth.npz"))
            result_inf_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ypsnr_model_weights.pth.npz"))
            result_inf_y = data["output_reco_array"]

            ssim_values_inf_z = np.load(
                os.path.join(path, f, "ssim_zpsnr_model_weights.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_z = np.load(os.path.join(path, f, "psnr_zpsnr_model_weights.pth.npy"))

            ssim_values_inf_y = np.load(
                os.path.join(path, f, "ssim_ypsnr_model_weights.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_y = np.load(os.path.join(path, f, "psnr_ypsnr_model_weights.pth.npy"))


            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_inf_z_last = data["output_reco_array"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_inf_y_last = data["output_reco_array"]

            ssim_values_inf_z_last = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_z_last = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_inf_y_last = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_inf_y_last = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

        elif "sigma_" + str(sigma) in f and "Inf" not in f and "Sob" not in f:
            data = np.load(os.path.join(path, f, "output_reco_results_zpsnr_model_weights.pth.npz"))
            result_z = data["output_reco_array"]
            clean = data["clean_test"]
            noisier = data["recos_test_z"]
            noisy = data["recos_test_y"]
            data = np.load(os.path.join(path, f, "output_reco_results_ypsnr_model_weights.pth.npz"))
            result_y = data["output_reco_array"]

            ssim_values_z = np.load(
                os.path.join(path, f, "ssim_zpsnr_model_weights.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_z = np.load(os.path.join(path, f, "psnr_zpsnr_model_weights.pth.npy"))

            ssim_values_y = np.load(
                os.path.join(path, f, "ssim_ypsnr_model_weights.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_y = np.load(os.path.join(path, f, "psnr_ypsnr_model_weights.pth.npy"))


            data = np.load(os.path.join(path, f, "output_reco_results_zlast.pth.npz"))
            result_z_last = data["output_reco_array"]
            data = np.load(os.path.join(path, f, "output_reco_results_ylast.pth.npz"))
            result_y_last = data["output_reco_array"]

            ssim_values_z_last = np.load(
                os.path.join(path, f, "ssim_zlast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_z_last = np.load(os.path.join(path, f, "psnr_zlast.pth.npy"))

            ssim_values_y_last = np.load(
                os.path.join(path, f, "ssim_ylast.pth.npy")
            )  # Changed ssim to ssim_values
            psnr_values_y_last = np.load(os.path.join(path, f, "psnr_ylast.pth.npy"))

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




    # Set up figure size and adjust spacing
    plt.figure(figsize=(10, 10))

    # Updated data, labels, and colors (N2I removed)
    psnr_data = [
        psnr_values_input,  # Input data
        psnr_values_sob_z, psnr_values_sob_z_last,  # Sob z and its variant
        psnr_values_sob_y, psnr_values_sob_y_last,  # Sob y and its variant
        psnr_values_inf_z, psnr_values_inf_z_last,  # Inf z and its variant
        psnr_values_inf_y, psnr_values_inf_y_last,  # Inf y and its variant
        psnr_values_z, psnr_values_z_last,          # z and its variant
        psnr_values_y, psnr_values_y_last,          # y and its variant
    ]
    psnr_labels = [
        'input', "ours sobo z (psnr)", "ours sobo z", "ours sobo y (psnr)", "ours sobo y", 
        "ours z (psnr)", "ours z", "ours y (psnr)", "ours y", 
        "z (psnr)", "z", "y (psnr)", "y"
    ]
    colors_psnr = ['#D3D3D3'] + ['#AEC6CF', '#FFB6C1'] * 6  # Gray for input, alternating colors for others

    # Plot each method
    for i in range(len(psnr_data)):
        plt.boxplot(psnr_data[i], positions=[i + 1], widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor=colors_psnr[i], color='blue'),
                    medianprops=dict(color='darkblue', linewidth=2),
                    whiskerprops=dict(color='blue', linestyle='--'))

    # Legend
    handles = [
        plt.Line2D([0], [0], color='#AEC6CF', lw=4, label='psnr'),
        plt.Line2D([0], [0], color='#FFB6C1', lw=4, label='last'),
        plt.Line2D([0], [0], color='#D3D3D3', lw=4, label='input'),
    ]

    # Titles and labels
    plt.legend(handles=handles, loc='upper right', fontsize=19)
    plt.title(f"PSNR Values (σ = {sigma})", fontsize=22, weight='bold')
    plt.xticks(ticks=range(1, len(psnr_labels) + 1), labels=psnr_labels, rotation=90, fontsize=19)
    plt.yticks(fontsize = 19)
    plt.xlabel("Methods", fontsize=22)
    plt.ylabel("PSNR (dB)", fontsize=22)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout(pad=2.0)
    plt.savefig(
        os.path.join(
            args.outputdir + "/Plots_Paper",
            "BoxPlot_ssim_psnr_" + str(sigma) + ".svg",
        )
    )
    plt.close()














