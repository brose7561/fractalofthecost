#!/bin/bash
#SBATCH --job-name=unet_train_4743787
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err



# -----------------------------
# Setup environment
# -----------------------------

source ~/miniconda/etc/profile.d/conda.sh
conda activate pytorch-env


# -----------------------------
# Navigate to project directory
# -----------------------------

cd $HOME/projects/demoTwo/Recognition/Unet

# -----------------------------
# Run training
# -----------------------------

srun python train.py 
