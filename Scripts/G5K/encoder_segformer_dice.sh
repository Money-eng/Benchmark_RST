#!/bin/bash

#OAR -q abaca
#OAR -p musa
#OAR -l host=1/gpu=2,walltime=6:00:00
#OAR -O run_segformer_dice.out
#OAR -E run_segformer_dice.err

source ~/.bashrc
mamba activate test
cd ~/Code
python3 ./RSA_deep_working/Models/main.py --config "./RSA_deep_working/Models/configs/segformer_dice.yml" > log_segformer_dice.log 2>&1
