
import argparse
import numpy as np
import os
import torch
from src.utils import *

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-1.7B")
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=8, help="Default max GPU memory in GiB")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--raw_dir", type=str, default = "./raw")
p.add_argument("--encoded_dir", type=str, default = "./encoded")
p.add_argument("--decoded_dir", type=str, default = "./decoded")
p.add_argument("--results_dir", type=str, default = "./result")
p.add_argument("--results_str", type=str, default = "result")
p.add_argument("--dataset_name", type=str, default = "tqa_qwen")
p.add_argument("--calculate_metric", type=int, default = 1)

args = p.parse_args()


if __name__ == "__main__":
    # Check if results_dir is exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    # Check if decoded_dir is exists
    if not os.path.exists(args.decoded_dir):
        os.makedirs(args.decoded_dir, exist_ok=True)

    # Read data from jsonl
    data =  load_testcases(DATASET_TO_PATH[args.dataset_name])

    # TODO: Assume that every layer is on GPU 0 now
    layer_to_device_id = {}
    for i in range(50):  # >=48, which is the num of layers for Qwen-14B
        layer_to_device_id[i] = 0
        
    average_acc = []
    total_raw_size = 0
    total_encoded_size = 0

    results = {}
    results['model'] = args.model_id
    results['dataset'] = args.dataset_name
    results['start'] = args.start
    results['end'] = args.end

    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    for doc_id in range(args.start, args.end):
        # Report the size of original KV cache, encoded KV cache, and decoded KV cache
        print(f"Processing KV Cache for doc_id: {doc_id}")
        original_kv_size = os.path.getsize(f"{args.raw_dir}/raw_kv_{doc_id}.pt")
        encoded_kv_size = os.path.getsize(f"{args.encoded_dir}/encoded_kv_{doc_id}.pt")   # Modify this line if you have modified encoded_file suffix
        decoded_kv_size = os.path.getsize(f"{args.decoded_dir}/decoded_kv_{doc_id}.pt")
        print("Original KV cache size: ", original_kv_size / 1024 / 1024, "MB")
        print("Encoded KV cache size: ", encoded_kv_size / 1024 / 1024, "MB")
        print("Decoded KV cache size: ", decoded_kv_size / 1024 / 1024, "MB")
        print("Compression Rate: ", encoded_kv_size / original_kv_size)
        total_raw_size += original_kv_size
        total_encoded_size += encoded_kv_size

        raw_kv = torch.load(f"{args.raw_dir}/raw_kv_{doc_id}.pt")
        decoded_kv = torch.load(f"{args.decoded_dir}/decoded_kv_{doc_id}.pt")
        print("Decoded KV cache shape: ", decoded_kv.shape)
        print("RMSE: ", rmse(raw_kv, decoded_kv))

        del raw_kv

        decoded_kv = decoded_kv.cuda()    # Comment this line if you don't have a GPU
        decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
        text = data[doc_id]['prompt']
        # if args.model_id == "Qwen/Qwen2.5-14B-Instruct":
        #     text = build_qwen2_prompt(tokenizer, text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        del decoded_kv
        del input_ids

        results[doc_id] = {
            'original_kv_size': original_kv_size / 1024 / 1024,
            'prediction': prediction.split('\n')[0]
        }

        if args.calculate_metric == 1:
            if args.dataset_name == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                print(f"Doc {doc_id} metric (F1 score): ", metric)
                results[doc_id]['f1_score'] = metric
                average_acc += [metric]
            elif args.dataset_name == "nqa" or args.dataset_name == "tqa" or args.dataset_name == "mqa" or args.dataset_name == "nqa_qwen" or args.dataset_name == "tqa_qwen" or args.dataset_name == "mqa_qwen":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                print(f"Doc {doc_id} metric (F1 score): ", metric)
                results[doc_id]['f1_score'] = metric
                average_acc += [metric]

        if args.dataset_name == "longchat":
            print(prediction, data[doc_id]['label'][0])
    if args.dataset_name == "longchat":
        metric_name = "accuracy"
    else:
        metric_name = "F1 score"
    if args.calculate_metric == 1:
        print(f"Average cachegen {metric_name} is: ", np.mean(average_acc))
        results['average_cachegen_acc'] = np.mean(average_acc)

    print("Average original KV cache size: ", total_raw_size / (args.end - args.start) / 1024 / 1024, "MB")
    print("Average compressed KV cache size: ", total_encoded_size / (args.end - args.start) / 1024 / 1024, "MB")
    print("Total compression rate: ", total_encoded_size / total_raw_size)

    results['average_original_kv_size'] = str(total_raw_size / (args.end - args.start) / 1024 / 1024) + " MB"
    results['average_compressed_kv_size'] = str(total_encoded_size / (args.end - args.start) / 1024 / 1024) + " MB"
    results['total_compression_rate'] = total_encoded_size / total_raw_size

    # Save results to json
    with open(f"{args.results_dir}/{args.results_str}.json", "w") as f:
        json.dump(results, f)