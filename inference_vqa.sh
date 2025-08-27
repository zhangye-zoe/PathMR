#!/bin/bash

# 设置环境变量 HF_HOME
export HF_HOME=/mnt/eternus/users/xx/project/iccv2025/hf_home/
export CUDA_VISIBLE_DEVICES=0

# 定义要使用的版本目录列表
versions=(
  "result_ckpts/PathVR_train_7b_v2"
  "result_ckpts/PathVR_train_13b_v1"
)

# 定义要依次使用的 JSON 文件列表
json_files=("shard_1" "shard_2")

for version in "${versions[@]}"; do
    # 获取版本名称，例如 LISA_finetune_7b 或 LISA_finetune_13b
    version_name=$(basename "$version")
    for json_file in "${json_files[@]}"; do
        echo "运行命令，使用 --val_json_name=${json_file} 版本: ${version_name}"
        python inference_vqa.py \
          --precision="bf16" \
          --version="${version}" \
          --output_jsonl_file="pathvr_v2/runs/${version_name}/vqa_data_test_${json_file}_answer.jsonl" \
          --test_json_file="data/vqa_data_test/${json_file}.json"
    
        echo "等待 30 秒..."
        sleep 30
    done
done
