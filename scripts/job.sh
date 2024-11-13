#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=1G
#SBATCH --job-name=train
#SBATCH --output=logs/train.out # specify a file to direct output stream
#SBATCH --error=logs/train.err
#SBATCH --open-mode=truncate # to overrides out and err files, you can also use


source /cluster/home/horatan/hand-fm/scripts/startup.sh
python train_mano_regressor.py
