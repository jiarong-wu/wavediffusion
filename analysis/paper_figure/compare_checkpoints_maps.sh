#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=m4874
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jw8736@nyu.edu

# Per-date map files of compare_checkpoints_stats.py's `maps` stage, then the
# figures. Resumable: re-submit after a timeout and finished dates are skipped.
#   sbatch compare_checkpoints_maps.sh 3                       # every OPTION3 checkpoint
#   sbatch compare_checkpoints_maps.sh 3 hist10d_ckpt14 ...    # only these
#   sbatch compare_checkpoints_maps.sh 3 --all-members ...     # also keep all members
module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
unset PYTHONPATH    # see compare_checkpoints.sh / CHANGELOG.md
OPTION=${1:-1}
python compare_checkpoints_stats.py $OPTION maps "${@:2}"
python compare_checkpoints_stats.py $OPTION plot_maps
for season in DJF MAM JJA SON; do
    python compare_checkpoints_stats.py $OPTION plot_maps $season
done
