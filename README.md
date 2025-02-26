# 📜 Noisier2Inverse: Self-supervised learning for one-step reconstruction of noisy inverse problems with correlated noise

*A deep learning framework for denoising and reconstructing medical images using self-supervised learning specifically designed for correlated noise.*



## 📂 Repository Structure  

Noisier2Inverse/  
│  
├── Data/                    # Data files (raw and processed)  
│   ├── raw/                 # Raw data (e.g., DICOM, CT scans)  
│   └── processed/           # Processed data (e.g., cleaned/augmented data)  
│  
├── Data_Heart/              # Documentation  
│   ├── index.md             # Main documentation file  
│   └── setup.md             # Setup instructions  
│  
├── bash_scripts/            # bash.sh is the bash script for running the noisier2inverse.py file, while the other bash scripts serve for plotting and reproducing the exact same plots from the paper  
│  
│── dataset.py               # Defines dataloaders  
│── model.py                 # Unet used as joint reconstruction and denoising method  
│── noisier2inverse.py       # Main script for running denoising  
│── two-step.py              # Two-step method described in our paper  
│── utils_inverse.py         # Necessary utils for noisier2inverse, helper functions  
│  
├── requirements.txt         # Required Python packages  
├── README.md                # Project description and setup instructions

## Workflow Diagram
![Noisier2Inverse Workflow](n2i_workflow.png)


## 🚀 Installation  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Nadja1611/Noisier2Inverse-Joint-Denoising-and-Reconstruction-of-correlated-noise.git
cd Noisier2Inverse
```
### **2️⃣ Create an environment and install LION** 

Install the LION package from [https://github.com/CambridgeCIA/LION](https://github.com/CambridgeCIA/LION)
```bash
cd ..
conda create -n noisier2inverse 
git clone https://github.com/CambridgeCIA/LION.git
cd LION
git submodule update --init --recursive
conda env create --file=env.yml
conda activate Lion
pip install .
```

### **3️⃣ Install all required packages of Noisier2Inverse using requirements file** 
As a last step, all additional packages required for Noisier2Inverse can be installed via the following command:

```bash
pip install -r requirements.txt
```


### **🚀 Run Noisier2Inverse**
Noisier2Inverse can be run by the following command

```bash
python noisier2inverse.py -l "DataDomain_MSE_EMD" --angles 512 --noise_type "gauss" --learning_rate 1e-5 --noise_intensity 1 --noise_sigma 5 --batch_size 4 --datadir '' --outputdir "" --weights_dir "" 
```

Now you're ready to run Noisier2Inverse and start denoising and reconstructing your images!

Let me know if you need any more tweaks! 😊
