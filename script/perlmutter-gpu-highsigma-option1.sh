#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --constraint=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gpus=4
#SBATCH --account=m4874
#SBATCH --job-name=highsigma1

# OPTION1 version of perlmutter-gpu-highsigma.sh: fine-tune OPTION1_periodic_hist10d
# ckpt 20 with the high-noise loss weighting, up to 6 epochs (~3.2 h per epoch);
# the __main__ of the .py resumes from the latest checkpoint up to 6 epochs
# (rerun until ckpt_6). Writes to scratch_folder/review/OPTION1_periodic_hist10d_highsigma/.
# Submit from this directory: cd ~/wavediffusion/script && sbatch perlmutter-gpu-highsigma-option1.sh
module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
# ~/.bashrc unconditionally exports PYTHONPATH pointing at ~/.local's site-packages
# (a different Python version's, no less), which leaks into every env on this
# system unless explicitly cleared here — see CHANGELOG.md.
unset PYTHONPATH

accelerate launch --multi_gpu ../example/experiments/meandiff_periodic_hist10d_highsigma.py
