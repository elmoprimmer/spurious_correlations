#!/bin/bash
#SBATCH --job-name=isic_dfr        # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/dfr_evaluate_spurious_%j.out  # Standard output path
#SBATCH --error=/home/primmere/logs/dfr_evaluate_spurious_%j.err   # Error output path
#SBATCH --ntasks=2                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # Total memory required per node
#SBATCH --time=03:00:00                   # Maximum runtime
#SBATCH --gpus=1                #

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2


python dfr_evaluate_spurious.py \
  --data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
  --result_path /home/primmere/logs/result_isic_6.pkl \
  --ckpt_path /home/primmere/logs/isic_logs_4/vit_isic_v2.pt \
  --skip_dfr_train_subset_tune True \
  --model_type "vit_b_16" \
  --batch_size 32 \
  --seed 7 \
  --DFR_retrained_model_path /home/primmere/logs/isic_logs_4/dfr_isic_model.pth \
  --DFR_logreg_path /home/primmere/logs/isic_logs_4/dfr_isic_logreg_2.pth \
  --label_csv /scratch_shared/primmere/isic/metadata_w_split_v2.csv \
  --dataset "isic"
