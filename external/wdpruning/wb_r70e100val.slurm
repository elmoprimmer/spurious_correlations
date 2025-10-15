#!/bin/bash
#SBATCH --job-name=wdpruning_wb
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/wdpruning/wb/r70e20val%j.out
#SBATCH --error=/home/primmere/logs/wdpruning/wb/r70e20val%j.err
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # memory per node
#SBATCH --time=06:00:00
#SBATCH --gpus=a100_80gb:1

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2


torchrun --standalone --nproc_per_node=1 \
    main_wdpruning.py \
    --arch vit_b_16_wb \
    --data-set wb \
    --batch-size 128 \
    --data-path /scratch_shared/primmere/waterbird \
    --output_dir logs/wb/r70e100val \
    --R_threshold 0.7 \
    --epochs 100
