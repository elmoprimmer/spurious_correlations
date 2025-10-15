#!/bin/bash
#SBATCH --job-name=pxp
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/pxp/isic/isic%j.out
#SBATCH --error=/home/primmere/logs/pxp/isic/isic%j.err
#SBATCH --ntasks=1                        # Number of tasks 
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # memory per node
#SBATCH --time=06:00:00
#SBATCH --gpus=1

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PARENT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
cd "$PARENT_DIR"

echo "SLURM script location:  $SCRIPT_DIR"
echo "Working directory (parent): $PARENT_DIR"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python wb_experiment.py   --layer-type Linear   --prune-r "0.6 0.7 0.8 0.9 0.92 0.94 0.96 0.98"   --n-indices 100   --groups "0 1 2 3"   --seed 3   --num-workers 8   --not_wb   --combining_relevances "3"
