#!/bin/bash
#SBATCH --job-name=pxp        # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/pxp/wb/wb%j.out
#SBATCH --error=/home/primmere/logs/pxp/wb/wb%j.err
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # memory per node
#SBATCH --time=06:00:00
#SBATCH --gpus=a100_80gb:1


module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2

PYTHONPATH=/home/primmere/ide/external/pruning_by_explaining \
python my_global_pruning.py \
--configs_path /home/primmere/ide/external/pruning_by_explaining/my_configs/test-config_waterbird7.yaml \
--output_path /home/primmere/logs/pxp/results/7/ \
--dataset_path /scratch_shared/primmere/waterbird \
--dataset_name waterbirds \
--num_workers 8




