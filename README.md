<h2 align="center"> PathVR: Multimodal Visual Reasoning for Interpretable Pathology Analysis </h2>

<p align="center">
  <a href="#news">News</a> |
  <a href="#abstract">Abstract</a> |
  <a href="#results">Results</a> |
  <a href="#installation">Installation</a> |
  <a href="#data">Data</a> |
  <a href="#checkpoints">Checkpoints</a> |
  <a href="#train">Train</a> |
  <a href="#inference">Inference</a>
</p>

## Innovations
## Tutorial

### Training
```bash
HF_HOME=/app/iccv2025/hf_home deepspeed \
 --include=localhost:4,5,6,7 \
 --master_port=24999 \
   train_ds.py \
 --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
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
 --exp_name="PathVR_train_v1" \
 --val_dataset="MultiPartReasonSeg|val" \
 --val_json_name="reason_seg_val.json"
```

### Deployment
```bash
CUDA_VISIBLE_DEVICES=0 python app.py \
--version='/app/iccv2025/PathVR/pathvr_v3/result_ckpts/ PathVR_v3_train_13b_v1' \
--precision='bf16'
```

##  Acknowledgement
This project is built upon [MMR](https://openreview.net/forum?id=mzL19kKE3r) and [SurgicalSAM](https://github.com/wenxi-yue/SurgicalSAM). The demo part is built upon [GLAMM](https://github.com/mbzuai-oryx/groundingLMM).  We thank the authors for their great work.
