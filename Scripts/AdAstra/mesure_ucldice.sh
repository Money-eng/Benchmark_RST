#!/bin/bash
#SBATCH --job-name=m_ucldice
#SBATCH --output=m_ucldice.out
#SBATCH --error=m_ucldice.err
#SBATCH --time=04:00:00
#SBATCH --account=cad16409

#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread

source ~/.bashrc
mamba activate utils

cd /lus/work/CT10/cad16409/lgandeel/

srun python ./RSA_reconstruction/main_measures.py --config /lus/work/CT10/cad16409/lgandeel/RSA_deep_working/Models/configs/unet_cldice.yml --path_to_results /lus/work/CT10/cad16409/lgandeel/Reconstruction/Unet_cldice > m_ucldice.log 2>&1
