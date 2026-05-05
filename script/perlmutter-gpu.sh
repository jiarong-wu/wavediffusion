#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

module purge

singularity exec --nv --overlay /scratch/jw8736/environments/mlflux.ext3:ro  \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
    -c "source /ext3/env.sh; python /scratch/jw8736/smalldiffusion/examples/wavediff.py"