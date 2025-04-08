#!/bin/bash
#SBATCH --job-name=train_isic_vit_w_split_v2         # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/train_isic_%j.out  # Standard output path
#SBATCH --error=/home/primmere/logs/train_isic_%j.err   # Error output path
#SBATCH --ntasks=2                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Total memory required per node
#SBATCH --time=06:00:00                   # Maximum runtime (6 hours)
#SBATCH --gpus=a100_80gb:1                # Request one A100 GPU with 80GB memory

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2


python train_classifier_isic.py \
  --data_dir "/scratch_shared/primmere/isic/isic_224/raw_224_with_selected" \
  --output_dir "/home/primmere/logs/isic_logs_4" \
  --eval_freq 10 \
  --seed 7 \
  --label_csv "/scratch_shared/primmere/isic/metadata_w_split_v2.csv" \
  --num_epochs 200 \
  --batch_size 32 \
  --pre_split True
