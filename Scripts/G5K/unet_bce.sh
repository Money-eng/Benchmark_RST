#!/bin/bash

#OAR -q abaca
#OAR -l host=1/gpu=2,walltime=6:00:00
#OAR -p musa
#OAR -O run_unet_bce.out
#OAR -E run_unet_bce.err

source ~/.bashrc
mamba activate test
cd ~/Code
python3 ./RSA_deep_working/Models/main_optuna.py --config "./RSA_deep_working/Models/configs/unet_bce.yml" > log_unet_bce.log 2>&1

