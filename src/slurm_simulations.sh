#!/bin/bash

#SBATCH --job-name=e_values
#SBATCH --output=logfolder/log_simulation_%A_%a.out
#SBATCH --error=logfolder/log_simulation_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition=normal,parietal
#SBATCH --array=0-299%2 # 2 * 5 * 30

# ---- Arrays ----
DATASETS=(0.0 3.0)

MODELS=(
    "lasso" "rf" "gb" "nn" "svr" #"lr" "lasso" "dt" "rf" "gb" "nn" "svr"
) # 15

# lr lasso dt rf et gb hgb ab bag mlp svr knn xgb SuperLearner TabICL
NUM_SEEDS=30

seed=$((1+SLURM_ARRAY_TASK_ID % NUM_SEEDS))

dataset_idx=$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))
model_idx=$((SLURM_ARRAY_TASK_ID % ${#MODELS[@]}))

corr=${DATASETS[$dataset_idx]}
mod=${MODELS[$model_idx]}

echo "Seed: $seed"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Beta strength: $corr"
echo "Model: $mod"

# ---- Run ----
python -m experiments.simulations \
    --seeds $seed \
    --beta_strength $corr \
    --model $mod

echo "Finished model=$mod beta_strength=$corr seed=$seed"