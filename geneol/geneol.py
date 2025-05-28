import numpy as np
from typing import Dict, List, Union, cast

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import json
import os
from peft import PeftModel, PeftConfig
from accelerate.utils import gather_object
import time

# Print CUDA environment info for debugging
print("[DEBUG] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG] torch.cuda.is_available():", torch.cuda.is_available())
print("[DEBUG] torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[DEBUG] torch.cuda.current_device():", torch.cuda.current_device())
    print("[DEBUG] torch.cuda.get_device_name():", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("[DEBUG] No CUDA device available!")

import openai
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import ast
import json_repair
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from scipy.spatial import ConvexHull
import re
from collections import defaultdict
from .openai_keys import *
from .prompts_utils import *



class LLM:

    def __init__(self, accelerator=None, args=None):
        self.args = args
        self.accelerator = accelerator
        self.model=None
        self.tokenizer=None
        self.ltokenizer=None
        
    # generate positives/negatives
    def generate(self, instructions_inputs_batch):
        if 'gpt' in self.args.gen_model_name_or_path:
            outputs=[]
            for messages in instructions_inputs_batch:
                retry_count = 0
                is_ok = False
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.args.gen_model_name_or_path,
                            messages=messages,
                            n=self.args.num_gens,
                            # temperature=args.temperature,
                            max_tokens=4096,
                            # stop=stop,
                            # top_p=args.top_p,
                        )
                        is_ok = True
                    except Exception as error:
                        time.sleep(1)
                        if retry_count % 5 ==0:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            continue


                outputs.extend([each_gen["message"]["content"] for each_gen in response["choices"]])
            return outputs
        else:
            
            embeddings = []
            batch_size = 32
            if(len(instructions_inputs_batch)>=batch_size):
                # import ipdb; ipdb.set_trace()
                # Assuming 'instructions_inputs_batch' is a list of inputs

                # batch_size = 4
                final_outputs = []
                final_embeddings = torch.tensor([]).cuda()

                # Split the instructions_inputs_batch into chunks of size 8
                for i in range(0, len(instructions_inputs_batch), batch_size):
                    batch = instructions_inputs_batch[i:i + batch_size]

                    # Tokenize and send to model
                    inputs = self.ltokenizer.apply_chat_template(
                        batch, padding=True, return_tensors="pt", return_dict=True
                    ).to("cuda")
                    
                    # Generate output
                    #do sameple = false 
                    output = self.model.generate(
                        **inputs, do_sample=False, num_return_sequences=self.args.num_gens, 
                        temperature=self.gtemp, repetition_penalty=1.0, 
                        top_p=self.top_p, max_new_tokens=1, 
                        return_dict_in_generate=True,
                        output_hidden_states=True # we want the hidden states to be accessible
                    )
                    
                    
                    # Process output
                    # output.sequences gives us [batch_size, sequence length]
                    #here we are selecting the entire batch, but only all the new tokens that are generated
                    # we can leave this for now since one extra token will be taken care of
                    new_output_ids = output.sequences[:, inputs.input_ids.shape[1]:]
                    
                    #output.hidden_states gives us a list of all the hidden states
                    #once we select the last layer with -1, then hidden_states[-1] has shape [batch_size, seq_len+1, hidden_dim]
                    #here we to select all batches, the last token in each sequence, and whole hidden dimension
                    #batch x dim
                    last_token_embeddings = output.hidden_states[-1][-1][:, -1, :]
                    
                    # print(last_token_embeddings, flush=True)
                    
                    
                    
                    del inputs; del output

                    #do we skip special tokens here???
                    outputs = self.tokenizer.batch_decode(
                        new_output_ids, skip_special_tokens=False, 
                        spaces_between_special_tokens=False
                    )
                    
                    # print(outputs, flush=True)
                    # print(last_token_embeddings.shape)
                    
                    del new_output_ids

                    # Clean outputs
                    outputs = [
                        o.replace("assistant\n\n", "", 1) if o.startswith("assistant\n\n") else o 
                        for o in outputs
                    ]
                    
                    # Append the processed outputs to the final list
                    final_outputs.extend(outputs)
                    final_embeddings = torch.cat([final_embeddings, last_token_embeddings.cuda()], dim=0)

                # Now final_outputs will contain all the processed outputs
                outputs=final_outputs
                embeddings = final_embeddings
                
                return outputs, embeddings

            else:    
                inputs = self.ltokenizer.apply_chat_template(instructions_inputs_batch, padding=True, truncation=True, max_length=3*self.args.max_length, return_tensors="pt", return_dict=True).to("cuda")
                
                output = self.model.generate(**inputs, do_sample=False, num_return_sequences=self.args.num_gens, temperature=self.gtemp, repetition_penalty=1.0, top_p=self.top_p, max_length=3*self.args.max_length, return_dict_in_generate=True, output_hidden_states=True)

                new_output_ids = output.sequences[:, inputs.input_ids.shape[1]:]
                
                # Get last token embeddings similar to the batch_size >= 4 case
                last_token_embeddings = output.hidden_states[-1][-1][:, -1, :]
                
                del inputs; del output
                outputs = self.tokenizer.batch_decode(new_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
                

                del new_output_ids
                outputs = [o.replace("assistant\n\n", "", 1) if o.startswith("assistant\n\n") else o for o in outputs]
                
                return outputs, last_token_embeddings
            
        # For GPT models, return outputs with None embeddings to maintain consistency
        return outputs, None

    # embedd using generations
    def embed(self, new_sentences_batch):
        # Convert list of dictionaries to properly formatted string
        if isinstance(new_sentences_batch, list) and isinstance(new_sentences_batch[0], dict):
            formatted_prompt = ""
            for msg in new_sentences_batch:
                formatted_prompt += f"{msg['role']}: {msg['content']}\n"
            new_sentences_batch = formatted_prompt

        inputs = self.tokenizer(new_sentences_batch, padding=True, truncation=True, return_tensors='pt',
                                max_length=4*self.args.max_length, add_special_tokens=False).to("cuda")

        all_embeddings = []
        batch_size = 1
        # Get the total number of sentences
        total_sentences = inputs['input_ids'].shape[0]

        # Iterate through the tokenized inputs in chunks of batch_size
        for i in range(0, total_sentences, batch_size):
            # Create a batch of tokenized inputs
            batch_inputs = {key: tensor[i:i+batch_size] for key, tensor in inputs.items()}
            
            # Get the embeddings based on whether the penultimate layer is used
            if self.args.penultimate_layer!=-1:
                hidden_states = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(
                    **batch_inputs, output_hidden_states=True, return_dict=True).hidden_states
                last_hidden_state = hidden_states[self.args.penultimate_layer]
            else:
                outputs = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**batch_inputs)
                last_hidden_state = outputs[0]
                del outputs
            
            # Append the batch embeddings to the list
            all_embeddings.append(last_hidden_state.detach().cpu())
        
        # Concatenate all the batch embeddings into one tensor
        final_embeddings = torch.cat(all_embeddings, dim=0)


        return final_embeddings, inputs

    def switchon_gen_model(self):
        k
        if self.args.method=='b5':
            return

        if 'gpt' in self.args.gen_model_name_or_path:
            import openai 
            

            OPENAI_API_BASE = "None"
            openai.api_key = OPENAI_API_KEY
            openai.organization = OPENAI_ORG_ID

            self.openaitokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False) # TODO: For ChatGPT we should use a different one
            self.model=None
        elif self.args.gen_model_name_or_path is not None:
            if self.model is not None: del self.model
            self.model = AutoModelForCausalLM.from_pretrained(self.args.gen_model_name_or_path, trust_remote_code=True, device_map={"": self.accelerator.process_index}, torch_dtype=self.args.torch_dtype)

            if('mistral' in self.args.gen_model_name_or_path.lower()):
                self.gtemp=0.7
                self.top_p=1
            elif('llama' in self.args.gen_model_name_or_path.lower()):
                self.gtemp=0.6
                self.top_p=0.9
            else:
                assert False, "Model not accepted"

            self.embedding_attr = 'model'
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f"Created {self.args.gen_model_name_or_path}: {self.model.dtype} dtype")


            self.ltokenizer = AutoTokenizer.from_pretrained(self.args.gen_model_name_or_path, padding_side='left')
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.gen_model_name_or_path, padding_side='right')
            if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.ltokenizer.pad_token = self.ltokenizer.eos_token

            print('Set pad token to eos token: ' + self.tokenizer.pad_token)
            self.model.eval()
        elif self.args.gen_model_name_or_path is None or self.args.gen_model_name_or_path == '-':
            pass
        else:
            pass

    def switchon_emb_model(self):       

        if self.model is not None: del self.model
        
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, trust_remote_code=True, device_map={"": self.accelerator.process_index}, torch_dtype=self.args.torch_dtype)

        if('mistral' in self.args.model_name_or_path.lower()):
            self.gtemp=0.7
            self.top_p=1
        elif('llama' in self.args.model_name_or_path.lower()):
            self.gtemp=0.6
            self.top_p=0.9
        else:
            assert False, "Model not accepted"

        self.embedding_attr = 'model'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Created {self.args.model_name_or_path}: {self.model.dtype} dtype")


        self.ltokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='right')

        # import ipdb; ipdb.set_trace()
        if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.ltokenizer.pad_token = self.ltokenizer.eos_token

        print('Set pad token to eos token: ' + self.tokenizer.pad_token)
        self.model.eval()

class GenEOL(torch.nn.Module):
    def __init__(self, accelerator, args) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.llm = LLM(accelerator, args)
        self.args=args
        self.pooling_method = args.pooling_method
        self.encode_call=0



    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)



    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        convert_to_tensor: bool = False,
        args=None,
        **kwargs,
    ) -> np.ndarray:
        self.encode_call+=1
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        print(args.task, flush=True)
        
        self.accelerator.wait_for_everyone()    
        start=time.time()
        
        all_embeddings = []
        with self.accelerator.split_between_processes(sentences) as sentences_rank_unchopped:
            np.random.seed(args.seed)
            self.llm.switchon_gen_model()
            
            save_path = os.path.join(args.output_folder, "..",'transformations')
            os.makedirs(save_path, exist_ok=True)
            print(save_path, flush=True)

            #this is a batch of strings that we are interested in processing
            sentences_rank = sentences_rank_unchopped
            
            
            #This is where we want to to run the mutiple generation-decode loop            
            num_generations = 5
            print(f"Number of generations  num_generations", flush=True)
            context = {}
            for gen_id in range(num_generations):
                iteration_embeddings = torch.tensor([]).cuda()
                print(f"\n\n\n\nIteration #{gen_id + 1}", flush=True)
                
                #Generate prompt for each of the setences
                all_prompts = []
                for sent_idx, sent in enumerate(tqdm(sentences_rank, desc="Encode-Decode Sentences", disable=len(sentences_rank)<50)):
                    
                    prompt = get_task_specific_emb_prompt(sent, args.task)
                    # print(f"\nInitial prompt: {prompt}", flush=True)
                    
                    context_words = context.get(sent_idx, [])
                    
                    
                    full_prompt = []
                    # if context_words:
                    #     for turn_idx, word in enumerate(context_words):
                    #         full_prompt.append({"role": "user", "content": prompt})
                    #         full_prompt.append({"role": "assistant", "content": word})
                    
                    if gen_id != 0:        
                        full_prompt.append({"role": "user", "content": f"Besides the components of {', '.join(context_words)} in the sentence, the setence can be distilled in one word" + prompt})
                    else:
                        full_prompt.append({"role": "user", "content": prompt})
                    
                    all_prompts.append(full_prompt)
                    
                    print(f"Full prompt with context: {full_prompt}", flush=True)
                    

                # Embed 
                #get the decoded output and the embedding
                #output: list of strings
                #embedding: [batch, dim]????
                output, generated_embedding_data = self.llm.generate(all_prompts)
                
                
                for x in range(0, len(output)):
                    if x not in context:
                        context[x] = []
                    context[x].append(output[x])
                    
                    # print(context, flush=True)
                
                iteration_embeddings = torch.cat([iteration_embeddings, generated_embedding_data], dim = 0)
                
                # #list of tensors of [batch_size, 1, hidden_dim]
                # if generated_embedding_data is not None:
                #     if isinstance(generated_embedding_data, list):
                #         # llm.generate returned a list of tensors
                #         for emb_tensor in generated_embedding_data:
                #             if torch.is_tensor(emb_tensor): # Ensure it's a tensor before .cpu()
                #                 all_embeddings.append(emb_tensor.cpu())
                #     elif torch.is_tensor(generated_embedding_data):
                #         # llm.generate returned a single tensor
                #         all_embeddings.append(generated_embedding_data.cpu())
                
                all_embeddings.append(iteration_embeddings)

        # all_embeddings = [all_embeddings]
        # all_embeddings=torch.cat(gather_object(all_embeddings), dim=0)
        
        stacked = torch.stack(all_embeddings, dim=0)
        print(f"Stacked tensors shape: {stacked.shape}", flush=True)
        all_embeddings = stacked.mean(dim=0)
        
        all_embeddings = all_embeddings if convert_to_tensor else all_embeddings.cpu().to(torch.float32).numpy()
        
        print(all_embeddings, flush=True)
        
        return all_embeddings
        
    def pooling(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [b, n, d]
            attention_mask: [b, n]
        """
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding
