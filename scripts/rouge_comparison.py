# Importing the required libraries
from rouge import Rouge
import numpy as np
import json

# Create a Rouge object
rouge = Rouge()

# Function to calculate ROUGE scores between a reference text and a list of candidate texts
def calculate_rouge_similarity(reference, candidates):
    rouge_scores = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": []
    }
    
    for candidate in candidates:
        try:
            if candidate=="":
                candidate=" "
            score = rouge.get_scores(candidate, reference)[0]
        except:
            import ipdb; ipdb.set_trace()
        rouge_scores["rouge-1"].append(score["rouge-1"]["f"])
        rouge_scores["rouge-2"].append(score["rouge-2"]["f"])
        rouge_scores["rouge-l"].append(score["rouge-l"]["f"])
    
    # Calculate mean scores for the window
    mean_rouge = {
        "rouge-1": np.mean(rouge_scores["rouge-1"]),
        "rouge-2": np.mean(rouge_scores["rouge-2"]),
        "rouge-l": np.mean(rouge_scores["rouge-l"]),
    }
    
    return mean_rouge

# Iterative comparison over windows
def iterative_comparison(items, window_size=32, assess_size=32):
    window_results = []
    for i in range(0, len(items), window_size + 1):
        reference = items[i]
        candidates = items[i + 1:i + assess_size + 1]
        if not candidates:  # If there are no candidates left, break the loop
            break
        mean_rouge = calculate_rouge_similarity(reference, candidates)
        window_results.append(mean_rouge)
    return window_results

# Function to calculate the overall mean ROUGE across all windows
def overall_mean_rouge(window_results):
    total_rouge = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": []
    }
    
    # Collect all individual window scores
    for window in window_results:
        total_rouge["rouge-1"].append(window["rouge-1"])
        total_rouge["rouge-2"].append(window["rouge-2"])
        total_rouge["rouge-l"].append(window["rouge-l"])
    
    # Calculate the final mean ROUGE scores across all windows
    final_mean_rouge = {
        "rouge-1": np.mean(total_rouge["rouge-1"]),
        "rouge-2": np.mean(total_rouge["rouge-2"]),
        "rouge-l": np.mean(total_rouge["rouge-l"]),
    }
    
    return final_mean_rouge


# Load JSON data
def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


# Path to the JSON file
json_file_path = "../resultsTF/1004_comp2_clus_s5_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_1_8/transformations/STS12_compositional_sentences1_0.json"
items = load_json(json_file_path)
window_results = iterative_comparison(items, 32)
final_rouge = overall_mean_rouge(window_results)
print(final_rouge)

json_file_path = "../resultsTF/1004_comp2_clus_s5_Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B_1_8/transformations/STS12_compositional_sentences1_0.json"
items = load_json(json_file_path)
window_results = iterative_comparison(items, 32)
final_rouge = overall_mean_rouge(window_results)
print(final_rouge)


# Load items from the JSON file

# Run the iterative comparison
