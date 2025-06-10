#!/bin/bash

#SBATCH --job-name=install_requirements
#SBATCH --account=paceship-efficient_geneol
#SBATCH --qos=inferno   
#SBATCH -C amd
#SBATCH --partition=cpu-md
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=./Nlogs/install_log.out
#SBATCH --error=./Nlogs/install_log.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@gatech.edu

# Load Anaconda module
module load anaconda3

# Create environment with specific Python version
conda create --name embeddings python=3.12 -y

# Activate environment
conda activate embeddings

conda install -y pip
# Install remaining requirements with no build isolation
pip install --upgrade cython
pip install -r /storage/home/hcoda1/4/pponnusamy7/p-efficient_geneol-0/GenEOL/requirements.txt

echo "Installation complete!"


