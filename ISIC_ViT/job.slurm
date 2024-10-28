#!/bin/bash
#SBATCH --job-name=train_isic_vit         # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/train_isic_%j.out  # Standard output path
#SBATCH --error=/home/primmere/logs/train_isic_%j.err   # Error output path
#SBATCH --ntasks=2                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Total memory required per node
#SBATCH --time=06:00:00                   # Maximum runtime (6 hours)
#SBATCH --gpus=a100_80gb:1                # Request one A100 GPU with 80GB memory

# Load necessary modules
module load cuda
module load miniconda


# Activate your conda environment
source $CONDA_ROOT/bin/activate
conda activate dfr2


# Execute your Python training script
python train_classifier_isic.py \
  --data_dir "/scratch_shared/primmere/isic/isic_224/raw_224" \
  --output_dir "/home/primmere/logs/isic_logs" \
  --eval_freq 10 \
  --seed 7 \
  --label_csv "/scratch_shared/primmere/isic/metadata.csv" \
  --num_epochs 100 \
  --batch_size 64
