#!/bin/bash

#OAR -q abaca
#OAR -p musa
#OAR -l host=1/gpu=2,walltime=6:00:00
#OAR -O run_unet_dice_cl_dice.out
#OAR -E run_unet_dice_cl_dice.err

source ~/.bashrc
mamba activate test
cd ~/Code
python3 ./RSA_deep_working/Models/main_optuna.py --config "./RSA_deep_working/Models/configs/unet_dice_cldice.yml" > log_unet_dice_cl_dice.log 2>&1
