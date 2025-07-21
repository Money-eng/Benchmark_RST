#!/bin/bash

#OAR -q abaca
#OAR -p musa
#OAR -l host=1/gpu=2,walltime=6:00:00
#OAR -O run_fsegformer_bce_dice.out
#OAR -E run_fsegformer_bce_dice.err

source ~/.bashrc
mamba activate test
cd ~/Code
python3 ./RSA_deep_working/Models/main_optuna.py --config "./RSA_deep_working/Models/configs/full_segformer_bce_dice.yml" > log_full_segformer_bce_dice.log 2>&1
