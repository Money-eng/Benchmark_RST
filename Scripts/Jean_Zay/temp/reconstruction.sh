#!/bin/bash
#SBATCH --job-name="u_bce"
#SBATCH -A vey@h100
#SBATCH -C h100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/rec_unet_bce.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/rec_unet_bce.err

module load arch/h100
module load pytorch-gpu/py3/2.7.0
module load openjdk/11.0.2 

cd /lustre/fswork/projects/rech/vey/unq35lq # $WORK

python3 ./RSA_reconstruction/main_reconstruction.py --config "./RSA_deep_working/Models/configs/unet_dice_cldice.yml" --model_path "./RSA_deep_working/Models/Checkpoints/Unet_cldice_dice/by_epochs/DataParallel_epoch060.pth" > /lustre/fswork/projects/rech/vey/unq35lq/log/rec_unet_dice_cldice.log 2>&1
