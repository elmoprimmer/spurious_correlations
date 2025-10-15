#!/bin/bash
#SBATCH --job-name=pxp
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/pxp/wb/wb%j.out
#SBATCH --error=/home/primmere/logs/pxp/wb/wb%j.err
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

python wb_experiment.py   --layer-type Softmax   --prune-r "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"   --n-indices 100   --groups "0 1 2 3"   --seed 1   --num-workers 8   --combining_relevances "2"
