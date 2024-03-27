#!/bin/bash
#SBATCH --signal=B:SIGTERM@120
#SBATCH -p gpu20
#SBATCH --mem=192G
#SBATCH --gres gpu:1
#SBATCH -o ./logs/slurm_evaluations/evalutate-%j.out
#SBATCH -t 8:00:00
#SBATCH --mail-type=fail
#SBATCH --mail-user=mrkristen@gmail.com

eval "$(conda shell.bash hook)"
  
conda activate EE3D_CVPR

export BATCH_SIZE=27
export TEMPORAL_STEPS="20"

python evaluate_ee3d_r.py
