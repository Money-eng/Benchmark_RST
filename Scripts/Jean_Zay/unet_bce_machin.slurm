#!/bin/bash
#SBATCH --job-name="u_bce"
#SBATCH -A vey@h100
#SBATCH -C h100
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/run_unet_bce_truc.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/run_unet_bce_truc.err

module load arch/h100

cd /lustre/fswork/projects/rech/vey/unq35lq # $WORK
source /lustre/fswork/projects/rech/vey/unq35lq/miniforge/bin/activate
mamba activate utils

# Dossier des checkpoints
CKPT_DIR="/home/lgandeel/Code/Results/Training/Checkpoints/Unet_bce/by_epochs/"
CONFIG="/home/lgandeel/Code/RSA_deep_working/Models/configs/unet_bce.yml"

# Lance jusqu'à 6 reconstructions en parallèle
ls -1 "${CKPT_DIR}"/*.pth \
| xargs -I{} -P 20 bash -c '
  ckpt="{}"
  base=$(basename "$ckpt" .pth)
  log="logs/${base}.log"
  mkdir -p logs
  echo "[START] $ckpt"
  python /home/lgandeel/Code/RSA_reconstruction/main_reconstruction.py --config "'"$CONFIG"'" --model_path "$ckpt" > "$log" 2>&1
  rc=$?
  echo "[END  ] $ckpt -> exit $rc (log: $log)"
'
