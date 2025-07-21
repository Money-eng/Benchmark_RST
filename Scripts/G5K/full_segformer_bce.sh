#!/bin/bash

#OAR -q abaca
#OAR -p musa
#OAR -l host=1/gpu=2,walltime=6:00:00
#OAR -O run_fsegformer_bce.out
#OAR -E run_fsegformer_bce.err

source ~/.bashrc
mamba activate test
cd ~/Code
python3 ./RSA_deep_working/Models/main_optuna.py --config "./RSA_deep_working/Models/configs/full_segformer_bce.yml" > log_full_segformer_bce.log 2>&1
