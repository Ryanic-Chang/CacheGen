import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
import json
import numpy as np
from collections import Counter
import re 
import string 
import pickle
import os

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# model_path = "/opt/downloaded_models/"
model_path = ""

dataset2metric = {
    "nqa": qa_f1_score,
    "nqa_qwen": qa_f1_score,
    "tqa": qa_f1_score,
    "tqa_qwen": qa_f1_score,
    "mqa": qa_f1_score,
    "mqa_qwen": qa_f1_score,
}

MAX_API_RETRY = 5
REQ_TIME_GAP = 2
DATASET_TO_PATH = {
    "longchat": "test_data/longchat.jsonl",
    "tqa": "test_data/tqa.jsonl",
    "nqa": "test_data/nqa.jsonl",
    "mqa": "test_data/mqa.jsonl",
    "tqa_qwen": "test_data/tqa_qwen.jsonl",
    "nqa_qwen": "test_data/nqa_qwen.jsonl",
    "mqa_qwen": "test_data/mqa_qwen.jsonl"
}

def scorer_e(dataset, predictions, answers, all_classes):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "tqa", "tqa_qwen", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        scores += [score]
    
    return scores

def to_blob(kv_tuples):
    """ Transform a list of tuples of key and value tensors to a single tensor
    """
    return torch.stack([torch.stack(inner_tuple, dim=0).to("cuda:0") for inner_tuple in kv_tuples], dim=0)

def calculate_acc(dataset_name, prediction, label):
    if dataset_name == "longchat":
        return []
    elif dataset_name == "nqa" or dataset_name == "nqa_qwen":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    elif dataset_name == "tqa" or dataset_name == "tqa_qwen":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    elif dataset_name == "mqa" or dataset_name == "mqa_qwen":
        scores = scorer_e(dataset_name, [prediction], [label['answers']], [label['all_classes']])
        return scores[0]
    
    
def define_model_and_tokenizer(model_id, num_gpus=1, max_gpu_memory=48):
    """ Define the model and tokenizer
    """
    if torch.cuda.is_available():
        device_map = "auto"
        max_memory = {0: "8GiB"}
    
    else:
        device_map = None
        max_memory = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path + model_id,
        # low_cpu_mem_usage=True,
        torch_dtype = torch.float16,
        device_map = device_map,
        max_memory = max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path + model_id)
            
    return model, tokenizer


def tensor_to_tuple(kv, layer_to_device_id):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}"), 
                       kv[i][1].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}")))
    return tuple(new_kv)


def load_testcases(test_file):
    with open(test_file, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases


def bw_generator(num_chunks):
    import numpy as np
    import random
    min = 0.1
    max = 10
    bw = np.zeros(num_chunks)
    for i in range(num_chunks):
        bw[i] = random.uniform(min, max)
    return bw

def profile(model, args):
    st = time.monotonic()
    input_ids = torch.randint(0, 32000, (1, args.chunk_size)).cuda()
    
    model.generate(input_ids,  do_sample=False,  max_new_tokens=1)
    torch.cuda.synchronize()
    return time.monotonic() - st


def bw_generator(num_chunks):
    import numpy as np
    import random
    min = 0.1
    max = 10
    bw = np.zeros(num_chunks)
    for i in range(num_chunks):
        bw[i] = random.uniform(min, max)
    return bw

def config_selection(all_bws, chunk_delay, args, length, doc_id):
    num_chunks = round(length / args.chunk_size)
    chunk_id = 0
    ttft = 0
    configs = []
    for chunk_start in range(0, length, args.chunk_size):
        bw = all_bws[chunk_id]
        found_cache = False
        
        for quant_level in np.arange(3, 0, -1):
            bytestream = pickle.load(open(f"{args.save_dir}/{doc_id}_{chunk_id}_{quant_level}.pkl", "rb"))
            if len(bytestream) / 1e9 * 8 / bw < args.slo / num_chunks:
                ttft += len(bytestream) / 1e9 * 8 / bw
                found_cache = True
                configs += [quant_level]
                break
        if not found_cache:
            ttft += chunk_delay
            configs += [0]
        chunk_id += 1
    return ttft, configs


def merge_kv(left, right, free_left = False, free_right = False):
    """
    Merges two kv caches, returns a merged KV cache
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - left: the left kv cache, could be None
    - right: the right kv cache

    Returns: The merged kv cache. If left is None, returns right
    """
    if left is None:
        return right
    #assert len(left) == len(right)

    def generator():
        for left_layer, right_layer in zip(left, right):
            yield (torch.cat([left_layer[0], right_layer[0]], dim = -2), torch.cat([left_layer[1], right_layer[1]], dim = -2))
            if free_left:
                del left_layer
            if free_right:
                del right_layer

    return tuple(generator())


def split_kv(kv, left: int, right: int):
    """
    Splits a kv cache into two kv caches
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - kv: the kv cache to be splitted
    - split_index: the index to split the kv cache

    Returns: a tuple of two kv caches
    """
    
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0][:, left:right].unsqueeze(0), 
                       kv[i][1][:, left:right].unsqueeze(0)))
    return tuple(new_kv)

# def merge(configs, args, doc_id, length, orig_kv = None, layer_to_device_id = None):
#     kv = []
#     chunk_id = 0
#     # simulation of the actual prefill
#     merged_kv = None
#     for chunk_start in range(0, length, args.chunk_size):
#         if chunk_start + args.chunk_size > length:
#             break
#         if configs[chunk_id] == 0:
#             loaded_kv = split_kv(orig_kv, chunk_start, chunk_start + args.chunk_size)
#         else:
#             os.environ["QUANT_LEVEL"] = str(configs[chunk_id])
#             loaded_kv = pickle.load(open(f"{args.save_dir}/{doc_id}_{chunk_id}_{configs[chunk_id]}.pkl", "rb"))
#             lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=args.chunk_size)
#             meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
#             deserializer = CacheGenDeserializer(lmcache_config, meta_data)
#             decoded_kv = deserializer.from_bytes(loaded_kv)
#             loaded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
#         merged_kv = merge_kv(merged_kv, loaded_kv)
#         chunk_id += 1
#     return merged_kv


def rmse(tensor1, tensor2):
    tensor1 = tensor1.float().to("cuda:0")
    tensor2 = tensor2.float().to("cuda:0")
    mse = torch.mean((tensor1 - tensor2)**2)
    rmse_val = torch.sqrt(mse)
    return rmse_val


def build_qwen2_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant giving helpful, detailed, and polite answers to the user's questions."},
        {"role": "user", "content": question}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)