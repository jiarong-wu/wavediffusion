#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=17:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m4874

module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
# ~/.bashrc unconditionally exports PYTHONPATH pointing at ~/.local's site-packages
# (a different Python version's, no less), which leaks into every env on this
# system unless explicitly cleared here — see CHANGELOG.md.
unset PYTHONPATH
accelerate launch --multi_gpu ../example/training/meandiff_periodic.py
# accelerate launch --multi_gpu ../example/experiments/meandiff_periodic_hist10d.py