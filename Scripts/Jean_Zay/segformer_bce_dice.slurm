#!/bin/bash
#SBATCH --job-name="lg_segformer_bce"
#SBATCH -A vey@h100
#SBATCH -C h100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/run_segformer_bce.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/run_segformer_bce.err

module load arch/h100
module load pytorch-gpu/py3/2.7.0

cd /lustre/fswork/projects/rech/vey/unq35lq # $WORK

python3 ./RSA_deep_working/Models/main.py --config "./RSA_deep_working/Models/configs/full_segformer_bce_dice.yml" > /lustre/fswork/projects/rech/vey/unq35lq/log/log_segformer_bce.log 2>&1
