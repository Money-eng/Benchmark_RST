#!/bin/bash

#OAR -q abaca
#OAR -l host=1/gpu=1,walltime=10:00:00
#OAR -p musa
#OAR -O /home/lgandeel/out/run_chrono.out
#OAR -E /home/lgandeel/err/run_chrono.err

source ~/.bashrc
mamba activate test
cd /home/lgandeel/Code/RSA_reconstruction/Method/ChronoRoot/

python3 /home/lgandeel/Code/RSA_reconstruction/Method/ChronoRoot/train_repro.py > /home/lgandeel/Code/RSA_reconstruction/Method/ChronoRoot/train_repro.log 2>&1
