#!/bin/bash
#SBATCH --signal=B:SIGTERM@120
#SBATCH -p gpu20
#SBATCH --mem=192G
#SBATCH --gres gpu:3
#SBATCH -o ./logs/slurm_outputs/slurm-%j.out
#SBATCH -t 30:00:00
#SBATCH --mail-type=fail
#SBATCH --mail-user=mrkristen@gmail.com

eval "$(conda shell.bash hook)"
  
conda activate EE3D_CVPR
                 
CHECKPOINT_PATH='' CUDA_VISIBLE_DEVICES=0,1,2 BATCH_SIZE=9 python train.py
