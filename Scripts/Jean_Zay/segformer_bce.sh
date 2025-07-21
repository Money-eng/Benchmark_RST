#!/bin/bash
#SBATCH --job-name="lg_segformer_bce"
#SBATCH -A vey@h100
#SBATCH -C h100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=36
#SBATCH --hint=nomultithread
#SBATCH --time=8:30:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/run_segformer_bce.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/run_segformer_bce.err

module load arch/h100
module load miniforge
mamba init
source ~/.bashrc
mamba activate test

cd /lustre/fswork/projects/rech/vey/unq35lq # $WORK

python3 ./RSA_deep_working/Models/main_optuna.py --config "./RSA_deep_working/Models/configs/segformer_bce.yml" > /lustre/fswork/projects/rech/vey/unq35lq/log/log_segformer_bce.log 2>&1
