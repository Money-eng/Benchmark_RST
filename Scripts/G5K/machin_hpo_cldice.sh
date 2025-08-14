#!/bin/bash

#OAR -q abaca
#OAR -l host=1/gpu=1,walltime=10:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_unet_cldice.out
#OAR -E /home/lgandeel/err/run_unet_cldice.err

source ~/.bashrc
mamba activate test
cd ~/Code

python3 ./RSA_deep_working/Models/hpo/run_hpo.py --config "./RSA_deep_working/Models/configs/unet_cldice.yml" > /home/lgandeel/log/machin_hpo_cldice.log 2>&1