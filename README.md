# KV260 MDE UNET

A project to train, quantize, compile and deploy a UNet model for deepth estimation on an AND Xilinx Kria KV260 MPSoC. It is mainly meant for people doing reasearch in the UMons Laboratory. This project rely heavily on previous work done in the laboratory by Nicolás Urbano Pintos (UTN FRH /CITEDEF) and Monal Patel Rakeshbhai (UMONS): https://github.com/nurbano/mde-unet-kv260


## Table of Contents

- [About](#about)
- [Requierments] (#requierments)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

This project is an internership reasearch project and is part of UMons research on edge computing vision alorithms. The goal is to allow easy deployement of depth estimation model on hardware target (AMD Xilinx Kria KV260). The project is mainly made of python scripts. It was made to target KV260 boards using Petalinux 2021.1 and Ubuntu 22.04.4. And uses DPU IPs loaded onto the KV260. The project is easier to use on a Linux host machine. The dataset used is a NYU Depth V2 split found on kaggle : https://www.kaggle.com/datasets/awsaf49/nyuv2-official-split-dataset


## Requirements

These are versions used for this project I do not have knowledge over how compatibility for these programms works :

- Docker version 29.6.2
- Conda 25.11.1


## Installation

[vitis-ai-version] should be chosen according to the Vitis-AI version on the KV260. We used version 1.4.1.978 for Petalinux 2021.1 and 2.5 for Ubuntu 22.04.4. If you are not using these version you will most probably encounter problems at the compilation steps that necessite ./compile.sh modifications.

[dataset-directory] should be the path to the dataset.

[python-version] should be the python version used for training the model (done outside the Vitis-AI docker environment). We used 3.11 in our case.

[configuration-file] should be a .yaml file on the model of ./configs/default.yaml (default.yaml can be used and modified).
IMPORTANT : While using it in Vitis-AI don't forget to change the Vitis-AI version parmeter

```bash
# Clone the repository
git clone https://github.com/madosaro/depth_estimation_kv260.git
cd depth_estimation_kv260

# Install dependencies
docker pull xilinx / vitis - ai :[vitis-ai-version]

# On linux an alias can be created for easier use of the Vitis-AI Docker
cat >> ~/.bashrc << 'EOF'
alias vitisai='docker run -it --rm --net=host \
  -v "$(pwd)":/workspace \
  -v [dataset-directory]:/workspace/dataset/ \
  -w /workspace \
  xilinx/[vitis-ai-version]'
EOF
source ~/.bashrc

# Vitis-ai Docker can be lauched by typing
vitisai

# Set-up conda environement
conda create --name myenv python=[python-version]
conda activate myenv

# Install python dependencies
pip install -r requirements.txt

```

## Usage

Parameters can be set in the ./configs config files and can then be used for the different commands

```bash
# Train a model (outside Vitis-AI docker to use GPU) -> outputs a training report
python3 train.py --config ./configs/[configuration-file]

# Quantize a model (should be launched inside Vitis-AI docker)
python3 quantization.py --config ./configs/[configuration-file]

# Evaluate the float, quantized and deployed (if the data has been collected) models. (should be launched inside Vitis-AI docker)
python3 evaluate.py --config ./configs/[configuration-file]

# Predict generate prediction depth map with the float, quantized and deployed (if the data has been collected) models. (should be launched inside Vitis-AI docker)
python3 predict.py --config ./configs/[configuration-file]
```
![Alt text](./models/MDE_UNET/outputs/training_report.png)
![Alt text](./models/MDE_UNET/outputs/deployment_2.5/visual_results/comparison_idx_10_rgb_00029.png
)
To use the model on the Kria Board, you should transfer : ./kria_petalinux or ./kria_ubuntu as well as the ./test split of the dataset with pairs of images and data used for testing.

You can then set-up the DPU we used the : kv260-dpu-benchmark for Petalinux and kv260-benchmark-b4096 for ubuntu. You can check the compatibility of the DPU by checking the target in the ./arch_*.json file you used to compile de model and the outpu of the xdputil querry command whilst a DPU is loaded.

```bash
# List all DPUs available (on the KV260)
sudo xmutil listapps

# Unload currently loaded DPU
sudo xmutil unloadapp

# Load a DPU
sudo xmutil loadapp [dpu-name]

# Check the DPU loaded
sudo xdputil querry
```

## Project Structure

```
DEPTH_ESTIMATION_KV260/
├── __pycache__/
├── .ipynb_checkpoints/
├── .vscode/
├── configs/
├── dataset/
├── ignore/
│   └── patate
├── kria_petalinux/
├── kria_ubuntu/
├── models/
│   └── MDE_UNET/
│       ├── build_1.4/
│       ├── build_2.5/
│       ├── checkpoints/
│       ├── outputs/
│       ├── MDE_UNET_history.json
│       ├── MDE_UNET.pth
│       └── TEST/
├── src/
│   ├── __pycache__/
│   ├── data/
│   ├── evaluation/
│   ├── losses/
│   ├── models/
│   ├── quantization/
│   ├── training/
│   ├── __init__.py
│   └── config.py
├── .gitignore
├── arch_1.4.json
├── arch_2.5.json
├── compile.sh
├── evaluate.py
├── predict.py
├── quantization.py
├── README.md
├── requirements.txt
└── train.py
```

## License

   This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.