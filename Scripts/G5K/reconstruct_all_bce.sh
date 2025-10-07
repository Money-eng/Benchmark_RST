#!/bin/bash
#OAR -q abaca
#OAR -l host=1/gpu=2,walltime=5:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_unet_bce.out
#OAR -E /home/lgandeel/err/run_unet_bce.err

sudo-g5k apt install -y xvfb

PYRUN="mamba run -n test python"

CONFIG="/home/lgandeel/Code/RSA_deep_working/Models/configs/unet_bce.yml"
CKPT_DIR="/home/lgandeel/Code/Results/Training/Checkpoints/Unet_bce/by_epochs"
SCRIPT="/home/lgandeel/Code/RSA_reconstruction/main_reconstruction.py"
LOGDIR="/home/lgandeel/Code/logs"

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
