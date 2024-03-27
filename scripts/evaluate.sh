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


# BEST VALIDATION 
# export EPOCH=34
# export BATCH_SIZE=32
# export NAME="CombinedEgoEvent_2023-10-28-12-33" 
# export EXP="baseline_3D_EROS"


# ## BEST TEST 
# export EPOCH=36
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-10-31-05-08" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"

# ## BEST TEST 
# export EPOCH=38
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-02-19-14" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="7"



# export EPOCH=40
# export BATCH_SIZE=32
# export NAME="CombinedEgoEvent_2023-10-28-12-33" 
# export EXP="baseline_3D_EROS"


# export EPOCH=34
# export BATCH_SIZE=32
# export NAME="CombinedEgoEvent_2023-10-28-12-49" 
# export EXP="baseline_3D"
# export TEMPORAL_STEPS="1"

# export EPOCH=40
# export BATCH_SIZE=32
# export NAME="CombinedEgoEvent_2023-10-28-12-49" 
# export EXP="baseline_3D"
# export TEMPORAL_STEPS="1"

# export EPOCH=35
# export BATCH_SIZE=32
# export NAME="CombinedEgoEvent_2023-10-28-12-49" 
# export EXP="baseline_3D"
# export TEMPORAL_STEPS="1"

## NEW models

# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-09-50" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"

# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-09-50" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="7"


# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-09-50" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="40"

# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-09-50" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="10"


# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-09-50" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="7"


# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-09-50" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="1"


export EPOCH=46
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2023-11-11-09-50" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="3"


export EPOCH=46
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2023-11-11-09-50" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="4"



# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-22-41" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"


# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-09-08-24" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"



# export EPOCH=47
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-14-05-35" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"


# export EPOCH=46
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-11-22-41"
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"

# export EPOCH=38
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-02-18-00"
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"


# export EPOCH=47
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-14-05-11"
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"


export EPOCH=47
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2023-11-16-21-35" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="20"


export EPOCH=48
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2023-12-22-16-03" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="20"

export EPOCH=47
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2023-11-14-05-11" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="20"


# export EPOCH=49
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-11-14-05-11" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"

# export EPOCH=49
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-12-25-13-01" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"

export EPOCH=50
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2023-12-26-15-34" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="20"


# CombinedEgoEvent_2023-11-14-05-11_2024-01-01-12-48

# export EPOCH=50
# export BATCH_SIZE=27
# export NAME="CombinedEgoEvent_2023-12-25-15-19" 
# export EXP="baseline_3D_EROS"
# export TEMPORAL_STEPS="20"


export EPOCH=51
export BATCH_SIZE=27
export NAME="CombinedEgoEvent_2024-01-09-15-05" 
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="20"


export EPOCH=4
export BATCH_SIZE=27
export NAME="EgoEvent_2023-11-19-11-04"
export EXP="baseline_3D_EROS"
export TEMPORAL_STEPS="20"



# CombinedEgoEvent_2023-12-25-15-19_epoch_49_checkpoint.pth

python evaluate.py
