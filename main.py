import argparse
import os
import time
import pickle
import torch
from src.utils import *


p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "Qwen/Qwen3-1.7B")
p.add_argument("--save_dir", type=str, default = "./raw")
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=8, help="Default max GPU memory in GiB")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--dataset_name", type=str, default="tqa_qwen")
args = p.parse_args()

if __name__ == "__main__":
    # Check if save_dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    model_device = model.device
    print("Model and tokenizer loaded on " + str(model_device))
    data = load_testcases(DATASET_TO_PATH[args.dataset_name])
    for doc_id in range(args.start, args.end):
        print("Saving KV cache for doc: ", doc_id)
        text = data[doc_id]['prompt']
        if args.model_id == "Qwen/Qwen2.5-14B-Instruct":
            text = build_qwen2_prompt(tokenizer, text)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model_device)

        st = time.monotonic()
        generated = model.generate(input_ids, max_new_tokens = 1, return_dict_in_generate=True)
        torch.cuda.synchronize()

        kv = generated['past_key_values']
        kv = list(kv)
        key_value = []
        for i in range(len(kv)):
            kv[i] = list(kv[i])
            kv[i][0] = kv[i][0][:, :, :-1][0]
            kv[i][1] = kv[i][1][:, :, :-1][0]
            kv[i] = tuple(kv[i])
        kv = tuple(kv)
        kv_tensor = to_blob(kv)
        
        torch.save(kv_tensor, f"{args.save_dir}/raw_kv_{doc_id}.pt")
        if doc_id == 0:
            pickle.dump(kv, open(f"{args.save_dir}/raw_kv_{doc_id}.pkl", "wb"))

        del kv
        del kv_tensor
        del input_ids
        del generated
        torch.cuda.empty_cache()

