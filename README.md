<h2 align="center"> PathMR: Multimodal Visual Reasoning for Interpretable Pathology Analysis </h2>
<p align="center">
  <a href="#Links">Links</a> |
  <a href="#-installation">Installation</a> |
  <a href="#training">Training</a> |
  <a href="#-deployment">Deployment</a> |
</p>




PathMR is a **cell-level multimodal visual reasoning framework** for pathological image analysis. It generates **expert-level diagnostic text explanations** and **pixel-level segmentation masks**, enhancing transparency and interpretability in AI-assisted pathology.

---

## 🔗 Links
- **Paper (arXiv preprint):** [https://arxiv.org/abs/xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)
- **Code:** [https://github.com/zhangye-zoe/PathMR](https://github.com/zhangye-zoe/PathMR)
- **Dataset (GADVR):** [https://github.com/zhangye-zoe/GADVR](https://github.com/zhangye-zoe/GADVR)

---

## 📦 Installation
1. Clone this repository
```bash
git clone https://github.com/zhangye-zoe/PathMR.git
cd PathMR

```

2. To install requirements using conda environment

```bash
conda env create -n [env name] -f M2SA.yaml
conda activate [env name]
pip install flash-attn --no-build-isolation
```

## 🏋️ Training
 
Below is an example command for training on **4 GPUs**:

```bash
deepspeed \
 --include=localhost:4,5,6,7 \
 --master_port=24999 \
 train_ds.py \
 --version="liuhaotian/llava-llama-2-13b-chat-lightning-preview" \
 --dataset_dir="data/GADVR" \
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
 --exp_name="PathMR" \
 --val_dataset="MultiPartReasonSeg|val" \
 --val_json_name="reason_seg_val.json"
```
When training is finished, to get the full model weight:

```
cd ./runs/PathMR/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

#### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:

```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```

## 🚀 Deployment

To launch the **interactive Gradio demo** for inference, run:

```bash
CUDA_VISIBLE_DEVICES=0 python app.py \
  --version="/app/iccv2025/PathVR/pathvr_v3/result_ckpts/PathVR_v3_train_13b_v1" \
  --precision="bf16"
```

## 🙏 Acknowledgements

This project builds upon the excellent work of the following projects:

- [**MMR**](https://openreview.net/forum?id=mzL19kKE3r) – for foundational ideas in multimodal reasoning.  
- [**SurgicalSAM**](https://github.com/wenxi-yue/SurgicalSAM) – for inspiring segmentation modules and implementation details.  
- [**GLAMM**](https://github.com/mbzuai-oryx/groundingLMM) – for providing the base framework for the interactive demo.

We sincerely thank the authors and contributors of these projects for their valuable efforts and open-source contributions.
