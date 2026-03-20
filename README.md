## 环境配置

```
conda env create -f env.yaml
conda activate tensoroffloader
```

## 运行 Transformers 前准备
Linux bash 运行：
export HF_ENDPOINT="https://hf-mirror.com"
或 Windows Powershell 运行：
$env:HF_ENDPOINT="https://hf-mirror.com"


## Step 1：预填充阶段
```
python main.py --model_id Qwen/Qwen3-1.7B --save_dir ./raw --start 0 --end 50 --dataset_name tqa_qwen
```
测试数据集共50个，单条数据生成的 KV Cache 在 500MB 上下，由于数据量大，你可以选择性地生成若干个。
dataset_name 可选：tqa_qwen nqa_qwen mqa_qwen。
生成后的 KV Cache 位于 ./raw/raw_kv_{doc_id}.pt。
生成过程可能很慢，我们已经附上将部分生成好的 KV Cache 结果。

## Step 2：压缩阶段
你的压缩程序需要从文件系统读取刚刚生成的 raw KV cache 数据，如使用 torch.load()（C++ 中同样可以使用 libtorch）。接下来，你需要实现对 KV Cache 高维张量的压缩算法，最终保存在压缩产物：encoded/encoded_kv_{doc_id}.{}。你可以自行设计压缩产物的文件后缀名（也就是可以设计文件格式），可以依然使用 .pt，也可以用二进制 .bin 等。

## Step 3：解压阶段
你的解压程序需要读取压缩阶段生成的压缩产物，实现对应的解压算法，还原回尽可能接近原始的高维张量，并将张量存储到解压产物：decoded/decoded_kv_{doc_id}.pt。**形状必须与 raw 文件夹下的 .pt 文件保持一致。** 最终保存在解压产物：decoded/decoded_kv_{doc_id}.pt

## Step 4：生成阶段
```
python run_cachegen_from_decoded_kv.py --model_id Qwen/Qwen3-1.7B --start 0 --end 50 --num_gpus 1 --results_str result --results_dir ./tqa_result --dataset_name tqa_qwen --calculate_metric 1
```
我们将会输出你的压缩产物与原始KV cache的压缩比，近似性（RMSE）等参数，以及使用还原后的KV cache进行推理的prediction结果准确度（F1 score）
dataset_name 可选：tqa_qwen nqa_qwen mqa_qwen。请对应修改results_dir
