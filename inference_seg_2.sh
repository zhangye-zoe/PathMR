#!/bin/bash

export HF_HOME="/app/iccv2025/hf_home/"
# 定义要依次使用的 JSON 文件列表
json_files=("split_reason_seg_val.json")

for json_file in "${json_files[@]}"
do
    echo "运行命令，使用 --val_json_name=${json_file}"
    deepspeed --include=localhost:4,5 --master_port=24992 inference_seg.py \
      --version="/app/iccv2025/PathVR/pathvr_v3/result_ckpts/PathVR_train_7b_decoder_only_smooth_loss_v1" \
      --dataset_dir="/app/iccv2025/PathVR/data/" \
      --dataset="vqa||multi_part_reason_seg" \
      --vqa_data="vqa_data_train" \
      --vision-tower="openai/clip-vit-large-patch14" \
      --num_classes_per_sample=1 \
      --use_expand_question_list \
      --batch_size=6 \
      --steps_per_epoch=3125 \
      --grad_accumulation_steps=1 \
      --model_max_length=2048 \
      --sample_rates="1,3" \
      --exp_name="PathVR_train_7b_decoder_only_smooth_loss_v1" \
      --val_dataset="MultiPartReasonSeg|val" \
      --val_json_name="${json_file}"

    echo "等待 30 秒..."
    sleep 30
done
