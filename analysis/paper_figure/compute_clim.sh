#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=2:00:00
#SBATCH --constraint=cpu
#SBATCH --account=m4874
#SBATCH --job-name=compute_clim

module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
unset PYTHONPATH    # see CHANGELOG.md
python compute_clim.py ${1:-1}
