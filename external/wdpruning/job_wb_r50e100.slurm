#!/bin/bash
#SBATCH --job-name=wdpruning_wb
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/wdpruning/wb/wbr50e100%j.out
#SBATCH --error=/home/primmere/logs/wdpruning/wb/wbr50e100%j.err
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
    --output_dir logs/wb/r50e100 \
    --classifiers 6 8 10 \
    --R_threshold 0.5 \
    --epochs 100
