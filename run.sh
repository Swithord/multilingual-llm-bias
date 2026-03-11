#!/bin/bash
#SBATCH --account=rrg-annielee
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

module load python/3.11 opencv gcc cuda arrow
source ~/env/bin/activate

export HF_HUB_DISABLE_XET=1
export HF_HOME=~/scratch/.cache/huggingface

python query.py
