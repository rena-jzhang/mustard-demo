#!/bin/bash

# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# Make the installer executable
chmod +x miniconda.sh

# Run the installer
./miniconda.sh -b -p ~/miniconda3

# Add Miniconda to your PATH
export PATH=~/miniconda3/bin:$PATH

source ~/.bashrc

# Initialize Conda (optional, removes the need to manually source)
conda init

conda create -n muvi python=3.10.6

conda activate muvi

git clone https://github.com/rena-jzhang/mustard-demo.git

cd mustard-demo

pip install -r requirements.txt