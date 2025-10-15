#!/bin/bash
#SBATCH --job-name=wdpruning_reproducing         # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/accuracy/wb/wdpruning/wb%j.out
#SBATCH --error=/home/primmere/logs/accuracy/wb/wdpruning/wb%j.err
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=2                 # Number of CPU cores per task
#SBATCH --mem=32G                         # memory per node
#SBATCH --time=01:00:00

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2


python print_dataset_sizes.py \
--data_dir /scratch_shared/primmere/waterbird \
--dataset "waterbirds"