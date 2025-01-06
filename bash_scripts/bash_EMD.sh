#!/bin/bash
#SBATCH --job-name=hwang_heart
#SBATCH --partition=zen3_0512_a100x2  
#SBATCH --qos=zen3_0512_a100x2    
#SBATCH --account=p72515
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4    # Adjust CPU allocation
#SBATCH --output=/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs/output_%j.out   # Standard output file (with unique job ID)
#SBATCH --error=/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs/error_%j.err     # Standard error file (with unique job ID)
module load cuda/11.8.0-gcc-12.2.0-bplw5nu
source /home/fs72515/nadja_g/.bashrc
conda activate Lion
nvidia-smi
python method_datadomain_EMD.py -l "DataDomain_MSE_EMD" --angles 512 --noise_type "gauss" --learning_rate 1e-5 --noise_intensity 1 --noise_sigma 5 --batch_size 4 --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/Model_Weights/" 
#python method_datadomain_inference_EMD.py -l "DataDomain_MSE_Inference_EMD_Sobolev" --angles 128 --noise_type "gauss" --learning_rate 5e-6 --noise_intensity 5 --noise_sigma 5 --batch_size 4 --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_nuts_sparse/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_nuts_sparse/Model_Weights/" 

#python method_datadomain_inference_EMD.py -l "DataDomain_MSE_Inference_EMD" --angles 128 --noise_type "gauss" --learning_rate 1e-4 --noise_intensity 5 --noise_sigma 5 --batch_size 4 --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_Final/" --weights_dir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_Final/Model_Weights/" 
