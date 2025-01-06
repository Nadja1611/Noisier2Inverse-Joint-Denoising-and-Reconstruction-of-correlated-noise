#!/bin/bash
#SBATCH --job-name=hwang_heart
#SBATCH --partition=zen3_0512_a100x2     
#SBATCH --qos=zen3_0512_a100x2          
#SBATCH --account=p72515
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1    # Adjust CPU allocation
#SBATCH --output=/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs/output_%j.out   # Standard output file (with unique job ID)
#SBATCH --error=/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/logs/error_%j.err     # Standard error file (with unique job ID)
module load cuda/11.8.0-gcc-12.2.0-bplw5nu
source /home/fs72515/nadja_g/.bashrc
conda activate Lion
nvidia-smi

#python plot_results.py  --angles 512  --path2 '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noise2Inverse/Results_Noise2Inverse_Heart/' --outputdir "/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/"  --path '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/' --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/' --noise_type "gauss" --dataset 'Heart' --learning_rate 7e-5 --noise_sigma 2 --noise_intensity 5 --batch_size 4

python plot_results_sigma.py  --angles 512  --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_Final/' --noise_type "gauss" --dataset 'Nuts' --learning_rate 7e-5 --noise_sigma 2 --noise_intensity 5 --batch_size 4
python plot_results_sigma.py  --angles 512   --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_Final/' --noise_type "gauss" --dataset 'Nuts' --learning_rate 7e-5 --noise_sigma 3 --noise_intensity 5 --batch_size 4
python plot_results_sigma.py  --angles 512   --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Nuts_Final/' --noise_type "gauss" --dataset 'Nuts' --learning_rate 7e-5 --noise_sigma 5 --noise_intensity 5 --batch_size 4


python plot_results_sigma.py  --angles 512 --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/' --noise_type "gauss" --dataset 'Heart' --learning_rate 7e-5 --noise_sigma 2 --noise_intensity 1 --batch_size 4
python plot_results_sigma.py  --angles 512  --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/' --noise_type "gauss" --dataset 'Heart' --learning_rate 7e-5 --noise_sigma 3 --noise_intensity 1 --batch_size 4
python plot_results_sigma.py  --angles 512   --datadir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Data_Heart/' --outputdir '/gpfs/data/fs72515/nadja_g/Noisier2Inverse/Noisier2Inverse/Results_Paper_Heart_Final/' --noise_type "gauss" --dataset 'Heart' --learning_rate 7e-5 --noise_sigma 5 --noise_intensity 1 --batch_size 4