#!/bin/bash

# 定义要依次使用的 JSON 文件列表
json_files=("reason_seg_val.json")

for json_file in "${json_files[@]}"
do
    echo "运行命令，使用 --val_json_name=${json_file}"
    HF_HOME=/mnt/eternus/users/Yu/project/iccv2025/hf_home deepspeed --include=localhost:0 --master_port=24997 /mnt/eternus/users/Yu/project/iccv2025/PathVR/pathvr_v3/inference_seg.py \
      --version="result_ckpts/PathMR-PathGen" \
      --dataset_dir="/mnt/eternus/users/Ye/TCGA/" \
      --dataset="multi_part_reason_seg" \
      --multi_reason_seg_data="reason_seg_val" \
      --vision-tower="openai/clip-vit-large-patch14" \
      --num_classes_per_sample=1 \
      --use_expand_question_list \
      --batch_size=6 \
      --steps_per_epoch=3125 \
      --grad_accumulation_steps=1 \
      --model_max_length=2048 \
      --sample_rates="1" \
      --exp_name="PathMR-PathGen" \
      --val_dataset="MultiPartReasonSeg|val" \
      --val_json_name="${json_file}"

    echo "等待 30 秒..."
    sleep 30
done