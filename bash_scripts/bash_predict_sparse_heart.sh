#!/bin/bash
#SBATCH --job-name=sparse_predict
#SBATCH --partition=zen3_0512_a100x2 
#SBATCH --qos=zen3_0512_a100x2      
#SBATCH --account=p72515
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1    # Adjust CPU allocation
#SBATCH --output=/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs/output_%j.out   # Standard output file (with unique job ID)
#SBATCH --error=/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs/error_%j.err     # Standard error file (with unique job ID)
module load cuda/11.8.0-gcc-12.2.0-bplw5nu
source /home/fs72515/nadja_g/.bashrc
conda activate Lion
nvidia-smi


python predict.py --angles 128 --noise_sigma 2 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_2.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py --angles 128 --noise_sigma 2 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_2.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py --angles 128 --noise_sigma 2 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_2.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_Sobolev_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 3 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_3.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 3 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_3.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 3 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_3.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_Sobolev_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 5 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_5.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 5 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_5.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 5 --noise_intensity 1 --dat 'z' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_5.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_Sobolev_alpha_1_learning_rate_1e-05_angles_128" 


python predict.py --angles 128 --noise_sigma 2 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_2.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py --angles 128 --noise_sigma 2 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_2.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py --angles 128 --noise_sigma 2 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_2.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_Sobolev_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 3 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_3.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 3 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_3.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 3 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_3.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_Sobolev_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 5 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_5.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 5 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_5.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_alpha_1_learning_rate_1e-05_angles_128" 
python predict.py  --angles 128 --noise_sigma 5 --noise_intensity 1 --dat 'y' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_sparse_Final/Model_Weights/Noise_gauss_sigma_5.0_1.0_batchsize_4_Gaussian_Method_DataDomain_MSE_Inference_EMD_Sobolev_alpha_1_learning_rate_1e-05_angles_128" 
