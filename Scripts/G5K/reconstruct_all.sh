#!/bin/bash

#OAR -q abaca
#OAR -l host=1/gpu=1,walltime=10:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_unet_bce_dice.out
#OAR -E /home/lgandeel/err/run_unet_bce_dice.err

source ~/.bashrc
mamba activate test
cd ~/Code
sudo-g5k apt install xvfb -y

ls -1 "${CKPT_DIR}"/*.pth \
| xargs -I{} -P 30 bash -c '
  ckpt="{}"
  base=$(basename "$ckpt" .pth)
  log="logs/${base}.log"
  mkdir -p logs
  echo "[START] $ckpt"
  python /home/lgandeel/Code/RSA_reconstruction/main_reconstruction.py --config "'"$CONFIG"'" --model_path "$ckpt" > "$log" 2>&1
  rc=$?
  echo "[END  ] $ckpt -> exit $rc (log: $log)"
'
echo "All reconstructions done."