#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=40:00:00
#SBATCH --constraint=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=4
#SBATCH --account=m4874

module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
# ~/.bashrc unconditionally exports PYTHONPATH pointing at ~/.local's site-packages
# (a different Python version's, no less), which leaks into every env on this
# system unless explicitly cleared here — see CHANGELOG.md.
unset PYTHONPATH

# accelerate launch --multi_gpu ../example/training/meandiff_periodic.py
# accelerate launch --multi_gpu ../example/experiments/meandiff_periodic_hist10d.py
# accelerate launch --multi_gpu ../example/experiments/wave_simple.py
accelerate launch --multi_gpu ../example/experiments/meandiff_periodic_hist10d_every2d.py
# accelerate launch --multi_gpu ../example/training/partdiff_periodic.py
# accelerate launch --multi_gpu ../example/experiments/partdiff_periodic_hist10d.py
# accelerate launch --multi_gpu ../example/experiments/partdiff_periodic_hist10d_det.py