# teacher-forcing evaluation for segmentation
import argparse
import os
import shutil
import sys
import csv
import time
from functools import partial

import logging
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import transformers
import copy

from peft import LoraConfig, get_peft_model

from model.PathVR import PathVRForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn

from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    AverageMeter,
    ProgressMeter,
    Summary,
    dict_to_cuda,
    intersectionAndUnionGPU,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="PathVR Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
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
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||multi_part_reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="2,9,2,6", type=str)  # Hyperparameter
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="vqa_data_train", type=str)
    parser.add_argument("--multi_reason_seg_data", default="reason_seg_train", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="/root/dataset/", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="test", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument(
        "--vision_pretrained",
        default="./vision_pretrained/sam_vit_h_4b8939.pth",
        type=str,
    )
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--num_classes_per_question", default=3, type=int)
    parser.add_argument("--use_expand_question_list", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--val_json_name", default="", type=str)

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:  # 0, 1, 2, 3, 4, 5, 6, 7
        os.makedirs(args.log_dir, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")

    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,  # train_mask_decoder: True
        "out_dim": args.out_dim,  # 256
        "ce_loss_weight": args.ce_loss_weight,  # 1.0, auto-regressive binary cross entropy loss for text generation
        "dice_loss_weight": args.dice_loss_weight,  # 0.5, segmentation loss
        "bce_loss_weight": args.bce_loss_weight,  # 2, segmentation loss
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,  # ViT-H
        "vision_tower": args.vision_tower,  # CLIP vision encoder of LLaVA model
        "use_mm_start_end": args.use_mm_start_end,  # True (SOS, EOS)
        "local_rank": args.local_rank,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # Declare the model
    model = PathVRForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    train_dataset = None

    val_dataset = ValDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        args.val_dataset,
        args.image_size,
        args.val_json_name,
    )

    print(f"Validating with {len(val_dataset)} examples for the seg.")

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
    }

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    result = validate_cell_type(val_loader, model_engine, args)


def validate_cell_type(val_loader, model_engine, args):
    overall_intersection = AverageMeter(
        "Overall_Intersection", fmt=":6.3f", summary_type=Summary.SUM
    )
    overall_union = AverageMeter("Overall_Union", fmt=":6.3f", summary_type=Summary.SUM)
    overall_iou = AverageMeter("Overall_IoU", fmt=":6.3f", summary_type=Summary.AVERAGE)

    model_engine.eval()
    category_metrics = {}

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        category_ids = input_dict["category_ids"]

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        category_ids = category_ids[0]
        assert len(pred_masks) == 1

        if len(output_list) == 0:
            continue
        for mask_i, output_i, cat in zip(masks_list, output_list, category_ids):
            intersection, union, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            fg_intersection = intersection[1].item()
            fg_union = union[1].item()
            # 当并集为 0（空目标）时定义 IoU 为 1.0
            fg_iou = 1.0 if fg_union == 0 else fg_intersection / (fg_union + 1e-5)

            overall_intersection.update(fg_intersection, n=1)
            overall_union.update(fg_union, n=1)
            overall_iou.update(fg_iou, n=1)

            cat_id = int(cat.item()) if torch.is_tensor(cat) else int(cat)
            if cat_id not in category_metrics:
                category_metrics[cat_id] = {
                    "intersection": AverageMeter(
                        f"Intersection_cat_{cat_id}",
                        fmt=":6.3f",
                        summary_type=Summary.SUM,
                    ),
                    "union": AverageMeter(
                        f"Union_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.SUM
                    ),
                    "iou": AverageMeter(
                        f"IoU_cat_{cat_id}", fmt=":6.3f", summary_type=Summary.AVERAGE
                    ),
                }

            category_metrics[cat_id]["intersection"].update(fg_intersection, n=1)
            category_metrics[cat_id]["union"].update(fg_union, n=1)
            category_metrics[cat_id]["iou"].update(fg_iou, n=1)

        # shape [2, 1]
        # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        # acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

        category_metrics[cat_id]["intersection"].update(fg_intersection, n=1)
        category_metrics[cat_id]["union"].update(fg_union, n=1)
        category_metrics[cat_id]["iou"].update(fg_iou, n=1)

    if dist.is_initialized():
        for cat_id in category_metrics:
            for meter in category_metrics[cat_id].values():
                meter.all_reduce()
        overall_intersection.all_reduce()
        overall_union.all_reduce()
        overall_iou.all_reduce()

    results = {}
    for cat_id, meters in category_metrics.items():
        inter_sum = meters["intersection"].sum
        union_sum = meters["union"].sum
        ciou = 1.0 if union_sum == 0 else inter_sum / (union_sum + 1e-10)
        giou = meters["iou"].avg
        results[cat_id] = {"ciou": ciou, "giou": giou}
        if args.local_rank == 0:
            print(f"Category {cat_id}: ciou: {ciou:.4f}, giou: {giou:.4f}")

    overall_ciou = (
        1.0
        if overall_union.sum == 0
        else overall_intersection.sum / (overall_union.sum + 1e-10)
    )
    overall_giou = overall_iou.avg

    if args.local_rank == 0:
        csv_file = args.val_json_name.replace("json", "csv")
        csv_file = f"{args.log_dir}/{csv_file}"
        # 将类别 id 按升序排列
        sorted_cat_ids = sorted(results.keys())
        header = (
            ["metric"] + [f"cat_{cat_id}" for cat_id in sorted_cat_ids] + ["overall"]
        )
        with open(csv_file, mode="w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            ciou_row = (
                ["ciou"]
                + [results[cat_id]["ciou"] for cat_id in sorted_cat_ids]
                + [overall_ciou]
            )
            csv_writer.writerow(ciou_row)
            giou_row = (
                ["giou"]
                + [results[cat_id]["giou"] for cat_id in sorted_cat_ids]
                + [overall_giou]
            )
            csv_writer.writerow(giou_row)
            print(f"Results saved to {csv_file}")
    results["overall"] = {"ciou": overall_ciou, "giou": overall_giou}

    return results


if __name__ == "__main__":
    main(sys.argv[1:])
