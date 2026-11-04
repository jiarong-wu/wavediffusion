#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=4
#SBATCH --account=m4874
#SBATCH --job-name=maps_period
#SBATCH --output=slurm-maps_period-%j.out

# compare_checkpoints_stats.py `maps` over another evaluation period, one
# (option, checkpoint) per GPU (at most 4), then the truth and checkpoint map
# figures. The truth files are written alongside the first run on each date.
#   EVAL_YEARS=2020-2025 EVAL_STRIDE_DAYS=5 sbatch compare_checkpoints_period.sh hist10d_det_ckpt20
#   OPTIONS=3 sbatch --time=06:00:00 compare_checkpoints_period.sh highsigma_ckpt5 resid_ckpt10
# Arguments: checkpoint names (plus --all-members), for each option in OPTIONS (default "1 3").
module load conda/Miniforge3-25.11.0-1
mamba activate wavediffusion
export PYTHONNOUSERSITE=1
unset PYTHONPATH    # see compare_checkpoints.sh / CHANGELOG.md
export EVAL_YEARS=${EVAL_YEARS:-2020-2025} EVAL_STRIDE_DAYS=${EVAL_STRIDE_DAYS:-5}
OPTIONS=${OPTIONS:-"1 3"}
FLAGS=(); NAMES=()
for a in "$@"; do [[ $a == --* ]] && FLAGS+=("$a") || NAMES+=("$a"); done
echo "Period $EVAL_YEARS every $EVAL_STRIDE_DAYS days; options $OPTIONS; maps ${NAMES[*]} ${FLAGS[*]}"

gpu=0
for o in $OPTIONS; do
    for name in "${NAMES[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu python compare_checkpoints_stats.py $o maps "${FLAGS[@]}" $name \
            > maps_period-$SLURM_JOB_ID-OPTION$o-$name.log 2>&1 &
        gpu=$(( (gpu + 1) % 4 ))
    done
done
wait
for o in $OPTIONS; do
    python compare_checkpoints_stats.py $o truth_maps
    for season in all DJF MAM JJA SON; do
        python compare_checkpoints_stats.py $o plot_maps $season
    done
done
