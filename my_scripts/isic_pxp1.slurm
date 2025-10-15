#!/bin/bash
#SBATCH --job-name=wdpruning_reproducing         # Job name
#SBATCH --partition=owner_fb12
#SBATCH --output=/home/primmere/logs/accuracy/wb/wdpruning/wb%j.out
#SBATCH --error=/home/primmere/logs/accuracy/wb/wdpruning/wb%j.err
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=32G                         # memory per node
#SBATCH --time=06:00:00
#SBATCH --gpus=1

module load cuda
module load miniconda
source $CONDA_ROOT/bin/activate
conda activate dfr2


python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--pruning_mask /home/primmere/logs/pxp/results/4/0.01.pth \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"

python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--pruning_mask /home/primmere/logs/pxp/results/4/0.05.pth \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"

python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--pruning_mask /home/primmere/logs/pxp/results/4/0.1.pth \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"

python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--pruning_mask /home/primmere/logs/pxp/results/4/0.2.pth \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"

python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--pruning_mask /home/primmere/logs/pxp/results/4/0.3.pth \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"

python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--pruning_mask /home/primmere/logs/pxp/results/4/0.5.pth \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"

python test_accuracy_w_groups.py \
--model_path '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt' \
--data_dir /scratch_shared/primmere/isic/isic_224/raw_224_with_selected \
--metadata_csv /scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv \
--pruning_mask /home/primmere/logs/pxp/results/4/0.8.pth \
--num_workers 8 \
--batch_size 32 \
--dataset "isic"


