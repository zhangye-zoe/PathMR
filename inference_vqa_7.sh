#!/bin/bash

export HF_HOME=/mnt/eternus/users/Yu/project/iccv2025/hf_home/
export CUDA_VISIBLE_DEVICES=6

versions=(
  "/mnt/eternus/users/Yu/project/iccv2025/PathVR/pathvr_v3/result_ckpts/PathVR_train_7b_decoder_weight_plus_smooth_loss_v2"
)

json_files=("shard_5" "shard_6")

for version in "${versions[@]}"; do
    version_name=$(basename "$version")
    for json_file in "${json_files[@]}"; do
        echo "run command, use --val_json_name=${json_file} version: ${version_name}"
        python inference_vqa.py \
          --precision="bf16" \
          --version="${version}" \
          --output_jsonl_file="/mnt/eternus/users/Yu/project/iccv2025/PathVR/pathvr_v3/runs/${version_name}/vqa_data_val_${json_file}_answer.jsonl" \
          --test_json_file="/mnt/eternus/users/Yu/project/iccv2025/PathVR/data/vqa_data_val/${json_file}.json"
    
        echo "wait 30 sec..."
        sleep 30
    done
done
