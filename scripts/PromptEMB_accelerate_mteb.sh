#!/bin/bash
#classification from 0-10
# datasets=(STS12 STS13 STS14 STS15 STS16 STS17 STS22 STSBenchmark SICK-R BIOSSES MR CR SUBJ MPQA SST2 TREC MRPC AmazonCounterfactualClassification Banking77Classification EmotionClassification MedrxivClusteringS2S TwentyNewsgroupsClustering AskUbuntuDupQuestions SciDocsRR StackOverflowDupQuestions TwitterSemEval2015 SprintDuplicateQuestions)
datasets=(STS12 STS13 STS14 STS15 STS16)
# datasets=(STS12 STS16)
              


convert_to_decimal() {
    local input_number=$1
    echo $((10#$input_number))
}
formatted_task_id=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})

# #! output file is used to extract the output folder name
# echo ${SLURM_OUTPUT_FILE}
#! job id is used to set the port number
job_id=$SLURM_JOB_ID
echo ${SLURM_JOB_ID}


index=$(convert_to_decimal $formatted_task_id)
suffix=$1
method=$2
gen_model=$3
emb_model=$4
batch_size=$5
num_gens=$6
shift 6
extras=$@
# index=$7
task=${datasets[index]}

# Calculate the job ID modulo 100
mod_job_id=$((job_id % 1000))
mod_job_id2=$(( (job_id % 10) * 100 ))
port=$((25500+index+mod_job_id+mod_job_id2))

# Define the array
echo "Dataset at index $index: $task: $model"
# accelerate launch --main_process_port ${port} eval_mteb_promptemb.py --model_name_or_path ${model} --attn cccc --task_names ${task} --instruction_set ${instruction_set} --instruction_format ${instruction_format} --batch_size ${batch_size} --num_gens ${num_gens} --suffix ${suffix}
accelerate launch --main_process_port ${port} eval_mteb_GenEOL.py --suffix ${suffix} --method ${method} --gen_model_name_or_path ${gen_model} --model_name_or_path ${emb_model}  --batch_size ${batch_size} --num_gens ${num_gens} --task_names ${task} ${extras}