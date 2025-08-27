import argparse
import os
import re
import sys

import bleach
import cv2
import random
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
from tools.markdown_utils import (
    markdown_default,
    examples,
    title,
    description,
    article,
    process_markdown,
    colors,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=2048, type=int)
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


tumor_keywords = [
    "neoplastic",
    "tumor",
    "cancer",
    "carcinoma",
    "malignant",
    "atypical",
]
inflammatory_keywords = [
    "inflatmmatory",
    "inflammatory",
    "immune",
    "leukocyte",
    "macrophage",
    "lymphocyte",
]
connective_keywords = [
    "connective",
    "fibroblast",
    "stromal",
    "stroma",
    "mesenchymal",
    "interstitial",
    "fibrous",
]
dead_keywords = ["dead", "necrotic", "apoptotic"]
epithelial_keywords = ["epithelial", "epithelium", "surface"]

mapping = {
    1: tumor_keywords,
    2: inflammatory_keywords,
    3: connective_keywords,
    4: dead_keywords,
    5: epithelial_keywords,
}


def predict_seg_token_category(predicted_text):
    """
    对预测文本 predicted_text 中所有的 [SEG] token 进行判断：
    检查 [SEG] 前最多三个单词（倒序优先检查），若出现合法关键词，则返回该关键词对应的类别 id，
    如果出现了多个关键词，则只取离 [SEG] 最近的关键词的类别 id。
    如果没有匹配到，则返回 None。

    返回一个列表，每个元素对应一个 [SEG] token 的类别 id。
    """
    tokens = predicted_text.split()
    seg_categories = []
    for i, token in enumerate(tokens):
        if token == "[SEG]":
            # 获取 [SEG] 前最多三个单词（不足三个则全部取）
            context_tokens = tokens[max(0, i - 3) : i]
            assigned_cat = None
            # 倒序遍历，先检查离 [SEG] 最近的单词
            for word in reversed(context_tokens):
                word_clean = re.sub(r"[^\w]", "", word).lower()
                for cat_id, keywords in mapping.items():
                    if word_clean in keywords:
                        assigned_cat = cat_id
                        break
                if assigned_cat is not None:
                    break
            seg_categories.append(assigned_cat)
    return seg_categories


color_map = {
    1: (0, 255, 0),  # 亮绿 (#00FF00)
    2: (0, 255, 255),  # 亮黄 (#FFFF00)
    3: (255, 255, 0),  # 青色 (#00FFFF)
    4: (0, 165, 255),  # 橙色 (#FFA500)
    5: (0, 0, 255),  # 亮红 (#FF0000)
}


def color_label(anno, label, color_map):
    mask = np.zeros((anno.shape[0], anno.shape[1], 3), dtype=np.uint8)
    mask[anno == 1] = color_map[label]
    return mask


def attach_anno(img, labels, annos, opacity=1.0):
    overlay = img.copy()
    assert len(labels) == len(annos), "Length of labels and annos should be the same."
    annos = [color_label(anno, label, color_map) for label, anno in zip(labels, annos)]
    for anno in annos:
        overlay = cv2.addWeighted(overlay, 1, anno, opacity, 0)
    return overlay


def wrap_seg_tokens(text):
    """
    按照规则：在每个 [SEG] token 前的两个单词添加 <p> 和 </p> 标签包裹。
    例如： "... brown fox [SEG] ..." → "... <p>brown fox</p> [SEG] ..."
    """
    # 这里使用正则表达式，匹配紧跟 [SEG] 前面的两个非空单词（假设单词间用空格分隔）
    pattern = r"(\S+\s+\S+)(\s*)(\[SEG\])"
    # 将匹配到的两个单词用 <p> 和 </p> 包裹后再保留原 [SEG]
    return re.sub(pattern, r"<p>\1</p>\2\3", text)


def prepare_mask(image_np, pred_masks, text_output, pred_classes):
    save_img = None
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue
        pred_mask = pred_mask.detach().cpu().numpy()
        mask_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
        if len(mask_list) > 0:
            save_img = image_np.copy()
            # colors_temp = predict_seg_token_category(text_output)
            seg_count = text_output.count("[SEG]")
            mask_list = mask_list[-seg_count:]
            for curr_mask, cls_color in zip(mask_list, pred_classes):
                color = color_map[cls_color]
                curr_mask = curr_mask > 0
                save_img[curr_mask] = (
                    image_np * 0.1
                    + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.9
                )[curr_mask]

    return save_img


args = parse_args(sys.argv[1:])
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
    **kwargs
)

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
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


# Gradio
examples = [
    [
        "How can epithelial cell nuclei be distinguished from surrounding components in an H&E stained image? Please output segmentation mask.",
        "data/image/val/21bc8d1891a242baaeb599847364e973_b3f058.jpg",
    ],
    [
        "Are inflammatory cells surrounding necrotic tumor regions indicating an inflammatory response present? Please respond with the segmentation mask.",
        "data/image/val/4b89d594689643aca85c1f49e44b602d_941012.jpg",
    ],
    [
        "Where are the areas with concentrated inflammatory cells located? Please output segmentation mask and explain why.",
        "data/image/val/9428cc34084f4c8f9516b2908d2f34a0_4d6c21.jpg",
    ],
    [
        "What features indicate vascular invasion in gastric adenocarcinoma on H&E staining? Please output segmentation mask.",
        "data/image/test/f1d5dfc0d13947b699e4cf710e99f895_551237.jpg",
    ],
]

# examples = [
#     [
#         "How do signet ring cells influence the spatial organization of stromal elements in gastric adenocarcinoma?",
#         "/app/iccv2025/PathVR/pathvr_v3/examples/image-1740999049693.png",
#     ],
#     [
#         "Is there evidence of a fibrous stroma with vascular presence in the tissue section?",
#         "/app/iccv2025/PathVR/pathvr_v3/examples/image-1740999049693.png",
#     ],
#     [
#         "What is the typical morphology of tumor cells in well-differentiated tubular adenocarcinoma?",
#         "/app/iccv2025/PathVR/pathvr_v3/examples/image-1740999049693.png",
#     ],
# ]
output_labels = ["Segmentation Output"]


## to be implemented
def inference(input_str, input_image):
    ## filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    # Model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = input_str
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = cv2.imread(input_image)
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
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    pattern = re.compile(r"ASSISTANT:\s*(.*?)\s*</s>", re.DOTALL)
    match = pattern.search(text_output)
    text_output = match.group(1)

    print("text_output: ", text_output)
    save_img = None
    if "[SEG]" in text_output:
        pred_classes = predict_seg_token_category(text_output)
        color_history = [color_map[cls] for cls in pred_classes]
        save_img = prepare_mask(image_np, pred_masks, text_output, pred_classes)
    print(save_img.shape if save_img is not None else "No seg output")
    output_str = text_output  # input_str
    if save_img is not None:
        output_image = save_img  # input_image
    else:
        ## no seg output
        output_image = input_image

    output_str = wrap_seg_tokens(output_str)
    markdown_out = process_markdown(output_str, color_history)
    # output_image = Image.fromarray(output_image)
    return output_image, markdown_out


demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Output Image"),
        gr.Markdown(label="Output Text"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging="auto",
    theme=gr.themes.Soft(),
)

demo.launch()
