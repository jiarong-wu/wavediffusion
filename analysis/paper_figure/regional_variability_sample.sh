#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=m4874
#SBATCH --job-name=regional_sample

# Regional means of model output on the dates in DATES_CSV (run the `dates` stage
# first). Resumable: re-submit after a timeout and finished dates are skipped.
module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
unset PYTHONPATH    # see CHANGELOG.md
python regional_variability.py sample ${@:-truth_resized hist10d_det_ckpt16 hist10d_ckpt16}
python regional_variability.py plot
