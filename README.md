# GenEOL Evaluation

  

This repository evaluates embeddings on the STS (Semantic Textual Similarity) datasets using GenEOL.

## Setup
Before running the evaluation, follow these steps:

#### Step 1: Add OpenAI Keys (Optional. Only required if you want to run openai generator models)
You need to add your OpenAI API keys to the project.
1. Create a file called `geneol/openai_keys.py`

```
OPENAI_API_KEY = ""
OPENAI_ORG_ID = ""
```
## Running the Evaluation

To run the code on STS datasets, use the following command:
```
accelerate launch --main_process_port 25502 eval_mteb_GenEOL.py \
--suffix 1015 \
--method s5 \
--gen_model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
--model_name_or_path mistralai/Mistral-7B-v0.1 \
--batch_size 2 \
--num_gens 2 \
--task_names STSBenchmark \
--compositional
```

### Command Explanation

-   `--gen_model_name_or_path mistralai/Mistral-7B-Instruct-v0.1`: Specifies the **generator model**.  Also accepts openai models - `gpt-3.5-turbo-0125`.
-   `--model_name_or_path mistralai/Mistral-7B-v0.1`: Specifies the **embedder model**. You can select a different embedder model similarly. Accepts mistral and llama family of models.
-   `--task_names STSBenchmark`: Specifies the dataset for evaluation. Options include:
    -   `STS12`, `STS13`, `STS14`, `STS15`, `STS16`, `SICK-R`
    For a full list of available datasets, refer to the `scripts/PromptEMB_accelerate_mteb` file.
-   `--suffix 1015`: This is a custom string used as a prefix for the output folder name where the results are saved. You can change it to any string value.
    