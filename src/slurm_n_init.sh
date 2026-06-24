#!/bin/bash

#SBATCH --job-name=e_values_n_init
#SBATCH --output=logfolder/log_simulation_%A_%a.out
#SBATCH --error=logfolder/log_simulation_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition=normal,parietal

# 4 n_init × 4 datasets × 1 model × 20 seeds = 320 jobs
#SBATCH --array=0-319%5

# ---- Arrays ----
N_INIT=(200 500 1000 2000)

DATASETS=(
    0.1
    0.3
    0.0
    0.6
)

MODELS=(
    "gb"
)

NUM_SEEDS=20

# ---- Decode task ID ----
task_id=$SLURM_ARRAY_TASK_ID

seed_idx=$((task_id % NUM_SEEDS))
task_id=$((task_id / NUM_SEEDS))

model_idx=$((task_id % ${#MODELS[@]}))
task_id=$((task_id / ${#MODELS[@]}))

dataset_idx=$((task_id % ${#DATASETS[@]}))
task_id=$((task_id / ${#DATASETS[@]}))

n_init_idx=$((task_id % ${#N_INIT[@]}))

# ---- Values ----
seed=$((10 + seed_idx))
corr=${DATASETS[$dataset_idx]}
mod=${MODELS[$model_idx]}
n_init=${N_INIT[$n_init_idx]}

echo "Seed: $seed"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "n_init: $n_init"
echo "Beta strength: $corr"
echo "Model: $mod"

# ---- Run ----
python -m experiments.simulations_n_init \
    --seeds "$seed" \
    --beta_strength "$corr" \
    --n_init "$n_init" \
    --model "$mod"

echo "Finished model=$mod beta_strength=$corr n_init=$n_init seed=$seed"