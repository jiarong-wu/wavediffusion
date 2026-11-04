#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --account=m4874
#SBATCH --job-name=partition_energy

# PDFs of partition energy / true total energy, truth vs sampled. Resumable:
# re-submit after a timeout and finished dates are skipped.
module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
unset PYTHONPATH    # see CHANGELOG.md
[ -f output/partition_energy_truth.npz ] || python partition_energy.py truth
python partition_energy.py sample ${@:-truth_resized hist5d_ckpt6 hist10d_ckpt4}
python partition_energy.py plot
