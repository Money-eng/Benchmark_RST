#!/bin/bash

#OAR -q abaca
#OAR -l host=1/gpu=1,walltime=15:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_full_segformer_cldice.out
#OAR -E /home/lgandeel/err/run_full_segformer_cldice.err

source ~/.bashrc
mamba activate test
cd ~/Code

python3 ./RSA_deep_working/Models/main.py --config "./RSA_deep_working/Models/configs/full_segformer_cldice.yml" > /home/lgandeel/log/full_segformer_cldice.log 2>&1
