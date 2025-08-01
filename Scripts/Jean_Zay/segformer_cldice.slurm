#!/bin/bash
#SBATCH --job-name="lg_segformer_cldice"
#SBATCH -A vey@h100
#SBATCH -C h100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/run_fsegformer_cldice.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/run_fsegformer_cldice.err

module load arch/h100
module load miniforge
mamba init
source ~/.bashrc
mamba activate utils

cd /lustre/fswork/projects/rech/vey/unq35lq # $WORK

python3 ./RSA_deep_working/Models/main_optuna.py --config "./RSA_deep_working/Models/configs/full_segformer_cldice.yml" > /lustre/fswork/projects/rech/vey/unq35lq/log/log_fsegformer_cldice.log 2>&1
