#!/bin/bash
#OAR -q abaca
#OAR -l host=1/gpu=2,walltime=5:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_segformer_bce_dice.out
#OAR -E /home/lgandeel/err/run_segformer_bce_dice.err

cd ~/Code
source ~/.bashrc
sudo-g5k apt install -y xvfb

mamba activate test
PYRUN="/home/lgandeel/miniforge3/envs/test/bin/python3"

CONFIG="/home/lgandeel/Code/RSA_deep_working/Models/configs/full_segformer_bce_dice.yml"
CKPT_DIR="/home/lgandeel/Code/Results/Training/Checkpoints/Segformer_bce_dice/by_epochs"
SCRIPT="/home/lgandeel/Code/RSA_reconstruction/main_reconstruction.py"
LOGDIR="/home/lgandeel/Code/logs/Segformer_bce_dice"

mkdir -p "$LOGDIR"

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -d "$CKPT_DIR" ]] || { echo "Checkpoint dir not found: $CKPT_DIR" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Script not found: $SCRIPT" >&2; exit 1; }

echo "Python used: $($PYRUN -c 'import sys;print(sys.executable)')"
echo "Torch ver  : $($PYRUN -c 'import torch;print(torch.__version__)')"

find "$CKPT_DIR" -maxdepth 1 -type f -name "*.pth" -print0 \
| xargs -0 -I{} -P 25 bash -c '
  ckpt="{}"
  base=$(basename "$ckpt" .pth)
  log="'"$LOGDIR"'/${base}.log"
  echo "[START] $ckpt"
  '"$PYRUN"' "'"$SCRIPT"'" --config "'"$CONFIG"'" --model_path "$ckpt" > "$log" 2>&1
  rc=$?
  echo "[END  ] $ckpt -> exit $rc (log: $log)"
'
echo "All reconstructions done."
