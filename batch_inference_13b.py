import argparse
import os
import sys
import json
import random
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from model.PathVR import PathVRForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def inference(
    prompt,
    image_path,
    model,
    tokenizer,
    clip_image_processor,
    transform,
    save_dir,
    task,
    args,
):
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if task != "vqa":
        prompt = prompt + "Please output the segmentation mask."
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    image_path = image_path.replace("/app", "/mnt/eternus/users/xx/project")

    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))
        return

    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

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

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=1024,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    if task != "vqa":
        for i, _pred_mask in enumerate(pred_masks):
            if _pred_mask.shape[0] == 0:
                continue
            for j, pred_mask in enumerate(_pred_mask):
                pred_mask = pred_mask.detach().cpu().numpy()
                pred_mask = (pred_mask > 0).astype(int)

                save_path = "{}/{}/{}_mask_{}.png".format(
                    save_dir, task, image_path.split("/")[-1].split(".")[0], j
                )
                try:
                    cv2.imwrite(save_path, pred_mask)
                except Exception as e:
                    print("Error in saving mask: ", e)
                    continue

    return text_output


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = PathVRForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    # define the dataset here:
    vqa_dataset = (
        "data/vqa_data_test.json"
    )
    refer_dataset = (
        "data/split_refer_seg_test.json"
    )
    reason_dataset = (
        "data/split_reason_seg_test.json"
    )

    save_dir = "./batch_result_13b"
    output_vqa = f"{save_dir}/vqa_result.jsonl"
    output_refer = f"{save_dir}/refer_result.jsonl"
    output_reason = f"{save_dir}/reason_result.jsonl"

    number = 200  # pick up 200 samples for each dataset
    # open the json file and retrive the prompt and image path
    with open(vqa_dataset, "r") as f, tqdm(total=number) as pbar, open(
        output_vqa, "w", encoding="utf-8"
    ) as out_f:
        datas = json.load(f)
        vqa_indices = generate_unique_indices(len(datas), number)
        task = "vqa"
        question_id = 0

        for index in vqa_indices:
            question_id += 1
            prompt = datas[index]["conversations"][0]["value"].strip("<image>\n")
            image_path = datas[index]["image"]
            answer = datas[index]["conversations"][1]["value"]

            pred = inference(
                prompt,
                image_path,
                model,
                tokenizer,
                clip_image_processor,
                transform,
                save_dir,
                task,
                args,
            )

            result = {
                "question_id": question_id,
                "question": prompt,
                "gt_answer": answer,
                "prediction": pred,
                "image_path": image_path.replace(
                    "/app", "/mnt/eternus/users/xx/project"
                ),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(1)

    with open(refer_dataset, "r") as f, tqdm(total=number) as pbar, open(
        output_refer, "w", encoding="utf-8"
    ) as out_f:
        datas = json.load(f)
        datas = datas["data"]
        task = "refer"
        question_id = 0
        refer_indices = generate_unique_indices(len(datas), number)
        for index in refer_indices:
            question_id += 1
            prompt = datas[index]["questions"][0]
            answer = datas[index]["text_answers"][0]
            image_path = datas[index]["img_origin"]
            ground_truth = os.path.join(
                "/mnt/eternus/users/Ye/Gastric/4. GADVR/test/overlay",
                datas[index]["file_name"],
            )
            pred = inference(
                prompt,
                image_path,
                model,
                tokenizer,
                clip_image_processor,
                transform,
                save_dir,
                task,
                args,
            )
            result = {
                "question_id": question_id,
                "question": prompt,
                "gt_answer": answer,
                "prediction": pred,
                "image_path": image_path.replace(
                    "/app", "/mnt/eternus/users/xx/project"
                ),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(1)

    with open(reason_dataset, "r") as f, tqdm(total=number) as pbar, open(
        output_reason, "w", encoding="utf-8"
    ) as out_f:
        datas = json.load(f)
        datas = datas["data"]
        task = "reason"
        question_id = 0
        reason_indices = generate_unique_indices(len(datas), number)
        for index in reason_indices:
            question_id += 1
            prompt = datas[index]["questions"][0]
            answer = datas[index]["text_answers"][0]
            image_path = datas[index]["img_origin"]
            ground_truth = os.path.join(
                "/mnt/eternus/users/Ye/Gastric/4. GADVR/test/overlay",
                datas[index]["file_name"],
            )
            pred = inference(
                prompt,
                image_path,
                model,
                tokenizer,
                clip_image_processor,
                transform,
                save_dir,
                task,
                args,
            )
            result = {
                "question_id": question_id,
                "question": prompt,
                "gt_answer": answer,
                "prediction": pred,
                "image_path": image_path.replace(
                    "/app", "/mnt/eternus/users/Yu/project"
                ),
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(1)


def generate_unique_indices(n, count=200, seed=42):
    random.seed(seed)
    indices = random.sample(range(0, n + 1), count)
    return indices


if __name__ == "__main__":
    main(sys.argv[1:])
