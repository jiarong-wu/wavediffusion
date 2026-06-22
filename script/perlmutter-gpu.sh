#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=18:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m4874

module load pytorch/2.6.0
accelerate launch --multi_gpu ../example/partdiff.py
