#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=m4874
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jw8736@nyu.edu

module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
# ~/.bashrc unconditionally exports PYTHONPATH pointing at ~/.local's site-packages
# (a different Python version's, no less), which leaks into every env on this
# system unless explicitly cleared here — see CHANGELOG.md.
unset PYTHONPATH
# OPTION (1 = mean wave variables, 3 = partitions): `sbatch compare_checkpoints.sh 3`
OPTION=${1:-1}
python compare_checkpoints_stats.py $OPTION
python plot_checkpoint_comparison.py $OPTION
