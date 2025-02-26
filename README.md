# 📜 Noisier2Inverse: Self-supervised learning for one-step reconstruction of noisy inverse problems with correlated noise

*A deep learning framework for denoising and reconstructing medical images using self-supervised learning specifically designed for correlated noise.*



---

## 📂 Repository Structure  

---
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

Noisier2Inverse can be run by the following command

```bash
python noisier2inverse.py -l "DataDomain_MSE_EMD" --angles 512 --noise_type "gauss" --learning_rate 1e-5 --noise_intensity 1 --noise_sigma 5 --batch_size 4 --datadir '' --outputdir "" --weights_dir "" 
```
