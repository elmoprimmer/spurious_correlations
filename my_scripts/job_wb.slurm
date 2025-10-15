#!/bin/bash
#SBATCH --job-name=wdpruning_reproducing         # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/accuracy/wb/wdpruning/wb%j.out
#SBATCH --error=/home/primmere/logs/accuracy/wb/wdpruning/wb%j.err
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
--model_path /home/primmere/ide/dfr/logs/vit_waterbirds.pth \
--data_dir /scratch_shared/primmere/waterbird \
--split "test" \
--num_workers 8 \
--batch_size 128 \
--dataset "waterbirds"