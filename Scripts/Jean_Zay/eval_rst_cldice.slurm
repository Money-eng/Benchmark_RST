#!/bin/bash
#SBATCH --job-name="rst_cldice"
#SBATCH -A vey@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/rst_cdice_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/rst_cdice_%j.err


module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
source /lustre/fswork/projects/rech/vey/unq35lq/envs/rst/bin/activate

cd $WORK

echo "Starting execution..."
srun --ntasks=1 python3 /lustre/fswork/projects/rech/vey/unq35lq/RSA_deep_working/Models/validate_and_evaluate.py \
    --config /lustre/fswork/projects/rech/vey/unq35lq/RSA_deep_working/Models/configs/unet_dice_cldice.yml \
    --weights /lustre/fswork/projects/rech/vey/unq35lq/checkpoints_rst/Unet_cldice_dice/by_epochs/ \
    > eval_rst_cdice.log 2>&1

echo "Execution finished."
