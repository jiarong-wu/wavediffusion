#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=4
#SBATCH --account=m4874
#SBATCH --job-name=highsigma

# Fine-tune OPTION3_periodic_hist10d ckpt 20 with the high-noise loss weighting,
# 3 epochs per job (~3.5 h each); the __main__ of the .py says whether it starts
# fresh or resumes (currently: resume ckpt_3 -> ckpt_4..6). Writes to scratch_folder/review/OPTION3_periodic_hist10d_highsigma/.
# Submit from this directory: cd ~/wavediffusion/script && sbatch perlmutter-gpu-highsigma.sh
module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
# ~/.bashrc unconditionally exports PYTHONPATH pointing at ~/.local's site-packages
# (a different Python version's, no less), which leaks into every env on this
# system unless explicitly cleared here — see CHANGELOG.md.
unset PYTHONPATH

accelerate launch --multi_gpu ../example/experiments/partdiff_periodic_hist10d_highsigma.py
