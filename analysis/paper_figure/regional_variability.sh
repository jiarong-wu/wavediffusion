#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --constraint=cpu
#SBATCH --account=m4874
#SBATCH --job-name=regional_variability

module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
unset PYTHONPATH    
export N_WORKERS=16
python regional_variability.py dates
python regional_variability.py compute ${@:-truth}
python regional_variability.py plot
