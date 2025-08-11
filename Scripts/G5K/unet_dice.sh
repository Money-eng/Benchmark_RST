#!/bin/bash

#OAR -q abaca
#OAR -l host=1/gpu=1,walltime=15:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_unet_dice.out
#OAR -E /home/lgandeel/err/run_unet_dice.err

source ~/.bashrc
mamba activate test
cd ~/Code

python3 ./RSA_deep_working/Models/main.py --config "./RSA_deep_working/Models/configs/unet_dice.yml" > /home/lgandeel/log/unet_dice.log 2>&1
