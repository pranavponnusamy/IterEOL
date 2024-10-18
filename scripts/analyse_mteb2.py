import json
import glob
import os
import numpy as np
import json_repair
import os
import re
import shutil

def filter_directories(path, include_pattern, exclude_patterns):
    """
    Iterates through directories in the specified path, keeps those matching the include pattern,
    and excludes those that match any of the exclude patterns. Returns a list of directories that are kept.
    
    Parameters:
        path (str): The directory path to search in.
        include_pattern (str): The regular expression pattern for directories to include.
        exclude_patterns (list): A list of regular expression patterns for directories to exclude.
    
    Returns:
        kept_dirs (list): A list of directories that are kept.
    """
    kept_dirs = []  # List to store kept directories
    
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            
            # Check if the directory matches the include pattern
            if re.search(include_pattern, dir_name):
                
                # Check if it matches any of the exclude patterns
                if not any(re.search(exclude_pattern, dir_name) for exclude_pattern in exclude_patterns):
                    # print(f"Keeping directory: {dir_path}")
                    kept_dirs.append(dir_path)  # Add to the kept directories list
                else:
                    pass
                    # print(f"Excluding directory: {dir_path}")
    
    return kept_dirs  # Return the list of kept directories

folder_list = filter_directories("../resultsTF/1018_comp2_s5_Mistral-7B-Instruct-v0.1_2_2/", ".*", ["transformations"])
folder_list = sorted(folder_list)
# folder_list = ["../resultsTF/0930_comp2_clus_s5_Mistral-7B-Instruct-v0.1_Mistral-7B-v0.1_2_2"]
# folder_list = ["../resultsTF/1009AB4_comp2_s5_Mistral-7B-Instruct-v0.1_1_8/"]
print(folder_list, flush=True)

# exit()
def is_json_path_present(json_obj, json_path):
    """
    Check if a JSON path exists inside a JSON object.

    Parameters:
    - json_obj: The JSON object to search.
    - json_path: The JSON path as a list of keys.

    Returns:
    - True if the JSON path exists, False otherwise.
    """
    current_obj = json_obj

    for key in json_path:
        if isinstance(current_obj, dict) and key in current_obj:
            current_obj = current_obj[key]
        else:
            return False

    return True








def get_sorted_values(input_dict):
    # Sort the dictionary by keys
    sorted_keys = sorted(input_dict.keys())

    # Return values in sorted order of keys
    return "\t".join(sorted_keys), "\t".join([str(input_dict[key]) for key in sorted_keys])

for folder in folder_list:
    main_score_list = {}
    classification_list = {}
    clustering_list = {}
    reranking_list = {}
    pairclassification_list = {}
    sts_list = {}
    retrieval_list={}
    summeval_list={}
    cqa_retrieval_list={}
    for json_file in glob.glob(os.path.join(folder, '*.json')):
        task_name = json_file.split("/")[-1].replace(".json", "")
        # print(task_name)
        with open(json_file, 'r') as file:
            try:
                # import ipdb; ipdb.set_trace()
                data = json_repair.load(file)
                # data = json.load(file)
            except:
                print(file)
                continue
            if 'scores' in data:
                data = data["scores"]
            mainscore=None
            # try:
            #     mainscore = data['test']['en']['main_score']
            #     classification_list[task_name] = mainscore
            # except:
            #     try:
            #         mainscore = data['test']['main_score']
            #         classification_list[task_name] = mainscore                   
            #     except:
            #         try:
            #             mainscore = data['test']['v_measure']
            #             clustering_list[task_name] = mainscore
            #         except:
            #             try:
            #                 mainscore = data['test']['cos_sim']['ap']
            #                 pairclassification_list[task_name] = mainscore
            #             except:
            #                 try:
            #                     mainscore = data['test']['map']
            #                     reranking_list[task_name] = mainscore
            #                 except:
            #                     try:
            #                         mainscore=data['test']['cos_sim']['spearman']
            #                         sts_list[task_name]=mainscore
            #                     except:
            #                         try:
            #                             mainscore = data['test']['en']['cos_sim']['spearman']
            #                             sts_list[task_name]=mainscore
            #                         except:
            #                             try:
            #                                 mainscore = data['test']['en-en']['cos_sim']['spearman']
            #                                 sts_list[task_name]=mainscore
            #                             except:
            if(is_json_path_present(data,['acc'])):
                mainscore = data['acc']/100
            elif(is_json_path_present(data,['test','en','main_score'])):
                mainscore = data['test']['en']['main_score']
                classification_list[task_name] = mainscore
                # print(f"test en {task_name} main_score")
            elif(is_json_path_present(data,['test','main_score'])):
                mainscore = data['test']['main_score']
                classification_list[task_name] = mainscore
                # print(f"test {task_name} main_score")
            elif(is_json_path_present(data,['test','v_measure'])):
                mainscore = data['test']['v_measure']
                clustering_list[task_name] = mainscore
                # print(f"test {task_name} v-measure")
            elif(is_json_path_present(data,['test','cos_sim','ap'])):
                mainscore = data['test']['cos_sim']['ap']
                pairclassification_list[task_name] = mainscore
                # print(f"test {task_name} cos-sim AP")            
            elif(is_json_path_present(data,['test','map'])):
                mainscore = data['test']['map']
                reranking_list[task_name] = mainscore
                # print(f"test {task_name} MAP")
            elif(is_json_path_present(data,['test','cos_sim','spearman']) and 'SummEval' in task_name):
                mainscore=data['test']['cos_sim']['spearman']
                summeval_list[task_name]=mainscore
                # print(f"test {task_name} cos-sim-spearman")
            elif(is_json_path_present(data,['test','cos_sim','spearman'])):
                mainscore=data['test']['cos_sim']['spearman']
                sts_list[task_name]=mainscore
                # print(f"test {task_name} cos-sim-spearman")
            elif(is_json_path_present(data,['test','en','cos_sim','spearman'])):
                mainscore = data['test']['en']['cos_sim']['spearman']
                sts_list[task_name]=mainscore
                # print(f"test en {task_name} cos-sim-spearman")
            elif(is_json_path_present(data,['test','en-en','cos_sim','spearman'])):
                mainscore = data['test']['en-en']['cos_sim']['spearman']
                sts_list[task_name]=mainscore
                # print(f"test en-en {task_name} ndcg_at_10")
            elif(is_json_path_present(data,['test','ndcg_at_10']) and 'CQA' in task_name):
                mainscore = data['test']['ndcg_at_10']
                cqa_retrieval_list[task_name]=mainscore
                # print(f"test {task_name} ndcg_at_10")
            elif(is_json_path_present(data,['test','ndcg_at_10'])):
                mainscore = data['test']['ndcg_at_10']
                retrieval_list[task_name]=mainscore
                # print(f"test {task_name} ndcg_at_10")
            elif(is_json_path_present(data,['dev','ndcg_at_10'])):
                mainscore = data['dev']['ndcg_at_10']
                retrieval_list[task_name]=mainscore
                # print(f"Dev {task_name} ndcg_at_10")
            else:
                # print(task_name, flush=True)
                mainscore=None
                mainscore = data['validation']['cos_sim']['spearman']
                sts_list[task_name]=mainscore
                # assert False, f"score not found {task_name}"
                # import ipdb; ipdb.set_trace()




            if mainscore is not None:
                # print(f'File: {json_file}, mainscore: {mainscore}')
                pass
            else:
                assert False
                # print(f'File: {json_file}, mainscore attribute not found.')

            if('CQA' not in task_name):
                main_score_list[task_name] = mainscore

    # retrieval_list["CQADupstackRetrieval"] = np.mean(list(cqa_retrieval_list.values()))
    # main_score_list["CQADupstackRetrieval"] = np.mean(list(cqa_retrieval_list.values()))
    # print(main_score_list)


    # main_score_list = {
    #     'AmazonCounterfactualClassification': 77.58,
    #     'AmazonPolarityClassification': 91.12,
    #     'AmazonReviewsClassification': 49.97,
    #     'ArguAna': 57.48,
    #     'ArxivClusteringP2P': 42.81,
    #     'ArxivClusteringS2S': 44.24,
    #     'AskUbuntuDupQuestions': 63.98,
    #     'BIOSSES': 85.24,
    #     'Banking77Classification': 88.31,
    #     'BiorxivClusteringP2P': 34.27,
    #     'BiorxivClusteringS2S': 35.53,
    #     'CQADupstackRetrieval': 48.84,
    #     'ClimateFEVER': 35.19,
    #     'DBPedia': 49.58,
    #     'EmotionClassification': 52.05,
    #     'FEVER': 89.40,
    #     'FiQA2018': 53.11,
    #     'HotpotQA': 74.07,
    #     'ImdbClassification': 87.42,
    #     'MSMARCO': 42.17,
    #     'MTOPDomainClassification': 96.04,
    #     'MTOPIntentClassification': 84.77,
    #     'MassiveIntentClassification': 79.29,
    #     'MassiveScenarioClassification': 81.64,
    #     'MedrxivClusteringP2P': 31.07,
    #     'MedrxivClusteringS2S': 31.27,
    #     'MindSmallReranking': 31.50,
    #     'NFCorpus': 39.33,
    #     'NQ': 61.70,
    #     'QuoraRetrieval': 87.75,
    #     'RedditClustering': 60.24,
    #     'RedditClusteringP2P': 64.12,
    #     'SCIDOCS': 22.50,
    #     'SICK-R': 83.70,
    #     'STS12': 78.80,
    #     'STS13': 86.37,
    #     'STS14': 84.04,
    #     'STS15': 88.99,
    #     'STS16': 87.22,
    #     'STS17': 90.19,
    #     'STS22': 67.68,
    #     'STSBenchmark': 88.65,
    #     'SciDocsRR': 83.80,
    #     'SciFact': 78.86,
    #     'SprintDuplicateQuestions': 96.82,
    #     'StackExchangeClustering': 70.73,
    #     'StackExchangeClusteringP2P': 34.50,
    #     'StackOverflowDupQuestions': 54.41,
    #     'SummEval': 29.96,
    #     'TRECCOVID': 77.69,
    #     'Touche2020': 22.18,
    #     'ToxicConversationsClassification': 69.26,
    #     'TweetSentimentExtractionClassification': 62.14,
    #     'TwentyNewsgroupsClustering': 52.18,
    #     'TwitterSemEval2015': 80.60,
    #     'TwitterURLCorpus': 86.56
    # }

    # print("-------------------------------------------------------------------------\n\n")
    print(folder, end="\t")
    # ord_list = ["AmazonCounterfactualClassification", "AmazonPolarityClassification", "AmazonReviewsClassification", "Banking77Classification", "EmotionClassification", "ImdbClassification", "MassiveIntentClassification", "MassiveScenarioClassification", "MTOPDomainClassification", "MTOPIntentClassification", "ToxicConversationsClassification", "TweetSentimentExtractionClassification", "AVG", "ArxivClusteringP2P", "ArxivClusteringS2S", "BiorxivClusteringP2P", "BiorxivClusteringS2S", "MedrxivClusteringP2P", "MedrxivClusteringS2S", "RedditClustering", "RedditClusteringP2P", "StackExchangeClustering", "StackExchangeClusteringP2P", "TwentyNewsgroupsClustering", "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus", "AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions", "ArguAna", "ClimateFEVER", "CQADupstackRetrieval", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact", "Touche2020", "TRECCOVID", "BIOSSES", "STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STS22", "STSBenchmark", "SICK-R", "SummEval"]
    # ord_list = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'AmazonCounterfactualClassification', 'Banking77Classification', 'EmotionClassification', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering', 'BiorxivClusteringS2S']
    ord_list = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R", "AVG", "STS17", "STS22", "BIOSSES"]
    # ord_list = ['AmazonCounterfactualClassification', 'Banking77Classification', 'EmotionClassification', 'MedrxivClusteringS2S', 'TwentyNewsgroupsClustering']


    for dname in ord_list:
        if(dname in main_score_list):
            print(main_score_list[dname]*100, end="\t")
        else:
            print("-", end="\t")
    print("")
    # print("\n\n-------------------------------------------------------------------------\n\n")
    # import ipdb; ipdb.set_trace()
    # print(f"{folder}\t{np.mean(list(classification_list.values()))}\t{get_sorted_values(classification_list)[1]}\t{np.mean(list(clustering_list.values()))}\t{get_sorted_values(clustering_list)[1]}\t{np.mean(list(reranking_list.values()))}\t{get_sorted_values(reranking_list)[1]}\t{np.mean(list(pairclassification_list.values()))}\t{get_sorted_values(pairclassification_list)[1]}\t{np.mean(list(sts_list.values()))}\t{get_sorted_values(sts_list)[1]}\t{np.mean(list(retrieval_list.values()))}\t{get_sorted_values(retrieval_list)[1]}\t{np.mean(list(summeval_list.values()))}\t{get_sorted_values(summeval_list)[1]}\t{np.mean(list(main_score_list.values()))}")
    # print(f"{folder}\t{np.mean(list(clustering_list.values()))}")
    # print(f"{folder}{get_sorted_values(pairclassification_list)[1]}\t{np.mean(list(sts_list.values()))}\t{get_sorted_values(sts_list)[1]}\t{np.mean(list(retrieval_list.values()))}\t{get_sorted_values(retrieval_list)[1]}\t{np.mean(list(summeval_list.values()))}\t{get_sorted_values(summeval_list)[1]}\t{np.mean(list(main_score_list.values()))}")
    # break;





