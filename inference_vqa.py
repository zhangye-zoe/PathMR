import argparse
import time
import json
import os
import re
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from model.PathVR import PathVRForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test JSON Inference for PathVR Model")
    parser.add_argument(
        "--test_json_file", type=str, required=True, help="测试 JSON 文件路径"
    )
    parser.add_argument(
        "--output_jsonl_file",
        type=str,
        default="inference_results.jsonl",
        help="推理结果保存文件",
    )
    parser.add_argument("--version", default="jdg900/PathVR-13B", help="模型版本")
    parser.add_argument(
        "--precision", default="bf16", choices=["fp32", "bf16", "fp16"], help="推理精度"
    )
    parser.add_argument("--image_size", type=int, default=1024, help="图像尺寸")
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="模型最大输入长度"
    )
    parser.add_argument(
        "--vis_save_path", type=str, default="./vis_output", help="可视化输出目录"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, help="生成最大 token 数"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="本地设备编号")
    parser.add_argument("--save_every_n", type=int, default=10, help="每 N 步保存一次")
    parser.add_argument(
        "--use_mm_start_end",
        action="store_true",
        default=True,
        help="是否使用 mm start/end 标记",
    )
    parser.add_argument(
        "--conv_type",
        type=str,
        default="llava_v1",
        choices=["llava_v1", "llava_llama_2"],
        help="对话模板类型",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    # 获取 [SEG] token 对应的 id
    seg_token_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = seg_token_id

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}

    # 加载模型
    model = PathVRForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower="openai/clip-vit-large-patch14",
        seg_token_idx=args.seg_token_idx,
        **kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        # vision_tower = model.get_model().get_vision_tower()
        # model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            # use_cuda_graph=True,
            use_triton=True,
            triton_autotune=True,
            # replace_with_kernel_inject=True,
        )
        model = model_engine.module

    return model, tokenizer


def estimate_remaining_time(start_time, processed, total):
    elapsed = time.time() - start_time
    if processed == 0:
        return "Estimating..."
    remaining_time = (elapsed / processed) * (total - processed)
    finish_time = time.localtime(time.time() + remaining_time)
    return f"剩余时间: {time.strftime('%H:%M:%S', time.gmtime(remaining_time))}, 预计完成时间: {time.strftime('%Y-%m-%d %H:%M:%S', finish_time)}"


def run_inference_for_sample(
    sample, model, tokenizer, clip_image_processor, args, question_id
):
    image = sample["image"].replace("/app", "/mnt/eternus/users/Yu/project/")
    question = sample["conversations"][0]["value"].strip("<image>\n")

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = question
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    if not os.path.exists(image):
        return None
    image_np = cv2.imread(image)
    if image_np is None:
        return None
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][
            0
        ]
        .unsqueeze(0)
        .cuda()
    )

    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = model.generate(
            images=image_clip,
            input_ids=input_ids,
            max_new_tokens=1024,
            num_beams=1,
            use_cache=False,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]

    return {"question_id": question_id, "prompt": question, "text": text_output}


def main():
    args = parse_args()
    os.makedirs(args.vis_save_path, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args)
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    with open(args.test_json_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    start_time = time.time()

    with open(args.output_jsonl_file, "w", encoding="utf-8") as out_f, tqdm(
        total=len(test_data), desc="推理进度"
    ) as pbar:
        for i, sample in enumerate(test_data):
            result = run_inference_for_sample(
                sample, model, tokenizer, clip_image_processor, args, question_id=i
            )
            if result:
                results.append(result)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (i + 1) % args.save_every_n == 0:
                out_f.flush()

            pbar.set_postfix_str(
                estimate_remaining_time(start_time, i + 1, len(test_data))
            )
            pbar.update(1)

    print(f"推理结果已保存到 {args.output_jsonl_file}")


if __name__ == "__main__":
    main()
