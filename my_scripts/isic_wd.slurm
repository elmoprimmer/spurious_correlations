#!/bin/bash
#SBATCH --job-name=wdpruning_reproducing         # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/accuracy/isic/wdpruning/isic%j.out
#SBATCH --error=/home/primmere/logs/accuracy/isic/wdpruning/isic%j.err
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # memory per node
#SBATCH --time=06:00:00
#SBATCH --gpus=a100_80gb:1

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2


python test_accuracy_w_groups.py \
--model_path /home/primmere/ide/external/wdpruning/logs/isic/r95e50val/checkpoint-best.pth \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--num_workers 16 \
--batch_size 64 \
--dataset "isic" \
--wdpruning True


