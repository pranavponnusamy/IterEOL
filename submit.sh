#!/bin/bash

#SBATCH --job-name=GenEOL_submission
#SBATCH --account=paceship-efficient_geneol
#SBATCH --qos=inferno
#SBATCH -C amd
#SBATCH --partition=cpu-amd
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=./Nlogs/test3/%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@gatech.edu

module load anaconda3
conda activate embeddings

# Just before you run your Python code:
export HF_HOME="./huggingface"
export HF_DATASETS_CACHE="./huggingface_datasets"
export HF_TOKEN="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"

# Array of select values
select_values=(8)

# Array of seed values
seed_values=(42)

# Models and their corresponding output subdirectories
declare -A models
models["mistralai/Mistral-7B-v0.3"]="mistral0.1"
# models["mistralai/Mistral-7B-Instruct-v0.1"]="mistral0.1"
# models["meta-llama/Meta-Llama-3-8B"]="llama3"
# models["Qwen/Qwen2.5-7B"]="qwen"
# models["google/gemma-3-12b-pt"]="gemma"

# Base directories and other parameters
base_output_dir="./Nlogs/r10-fullconetext-final"
base_script="./scripts/PromptEMB_accelerate_mteb.sh"
model_1="mistralai/Mistral-7B-Instruct-v0.1"
# model_1="mistralai/Mistral-7B-v0.3"
#model_1="google/gemma-3-12b-it"
task_name="r10-fullconetext-final"
session="s5"
gpu_count=1
task_per_node=1
account="paceship-efficient_geneol"

# SLURM parameters
partition="gpu-h100"
array="0-5%10"
gres="gpu:h100:1"
ntasks=1
mem="32G"

# Create output directory if it doesn't exist
mkdir -p "$base_output_dir"

# Outer loop: iterate over the models
for model_2 in "${!models[@]}"; do
  model_subdir="${models[$model_2]}"

  # Middle loop: iterate over the select values
  for select_value in "${select_values[@]}"; do

    # Inner loop: iterate over the seed values
    for seed in "${seed_values[@]}"; do
      # Include the seed value in the output directory
      output_dir="${base_output_dir}/${model_subdir}_k${select_value}_seed${seed}"
      mkdir -p "$output_dir"

      sbatch --account=$account \
             --qos=embers \
             --partition=gpu-h100 \
             --gres=gpu:h100:1 \
             --mem-per-gpu=80G \
             --time=2:00:00 \
             --array=0-5%10 \
             --output="${output_dir}/%A_%a.out" \
             --error="${output_dir}/%A_%a.err" \
             --mail-type=BEGIN,END,FAIL \
             --mail-user=$USER@gatech.edu \
             $base_script $task_name $session $model_1 $model_2 $gpu_count $task_per_node \
             --compositional --select $select_value --seed $seed
    done
  done
done
