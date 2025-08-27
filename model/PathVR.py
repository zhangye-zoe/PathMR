from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


from .llava.model.language_model.llava_llama import (
    LlavaLlamaForCausalLM,
    LlavaLlamaModel,
)
from .segment_anything import build_sam_vit_h

from .prompt_encoder import Prototype_Prompt_Encoder, Learnable_Prototypes


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss * weight
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def total_variation_loss(pred_mask: torch.Tensor):
    if pred_mask.shape[0] == 0:
        return 0

    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)  # 转换为 (B, 1, H, W)

    loss_h = torch.abs(pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]).mean()
    loss_w = torch.abs(pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]).mean()
    return loss_h + loss_w


def calculate_weights(masks: torch.Tensor, category_ids: torch.Tensor, lambda_weight=1):
    """
    Calculate the weights for each mask based on the category id
    """
    if len(masks) == 0:  # for vqa
        return torch.ones_like(masks)
    cat2idx = {}
    weights = []
    category_ids = category_ids.tolist()
    core_category = list(set(category_ids))
    for idx in core_category:
        cat2idx[idx] = [i for i, j in enumerate(category_ids) if j == idx]
    for i in range(len(masks)):
        weight = torch.ones_like(masks[i])
        for k, v in cat2idx.items():
            if k != category_ids[i]:
                weight += masks[v[0]] * lambda_weight
        weights.append(weight)
    weights = torch.stack(weights, dim=0)
    return weights


class PathVRMetaModel:
    def __init__(self, config, **kwargs):
        super(PathVRMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_PathVR_modules(self.config)

    def initialize_PathVR_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)

        for param in self.visual_model.parameters():
            param.requires_grad = False

        # Decoder training on
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.classification_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),
        )
        self.classification_head.train()
        for param in self.classification_head.parameters():
            param.requires_grad = True


class PathVRModel(PathVRMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PathVRModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class PathVRForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            self.local_rank = kwargs.pop("local_rank", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        # self.num_classes = kwargs.pop("num_classes")
        # assert self.num_classes == 6, "num_classes should be 6"

        super().__init__(config)

        self.model = PathVRModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Extract the visual embeddings from SAM-VIT-H
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            early_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings, early_embeddings = (
                    self.model.visual_model.image_encoder(pixel_values[i].unsqueeze(0))
                )
                image_embeddings_list.append(image_embeddings)
                early_embeddings_list.append(early_embeddings[0].permute(0, 3, 1, 2))
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
            early_embeddings = torch.cat(early_embeddings_list, 0)
        return image_embeddings, early_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):

        image_embeddings, early_embeddings = self.get_visual_embs(images)
        # image_embeddings.shape: [2, 256, 64, 64],
        # early_embeddings are from 8th block of ViT-H Image Encoder: [2, 1280, 64, 64]
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=(
                    image_embeddings[i].unsqueeze(0),
                    early_embeddings[i].unsqueeze(0),
                ),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )  # [# of tokens, 1, 256, 256]

            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        variation_loss = 0
        num_masks = 0
        lambda_tv = 0
        lambda_class_weight = 0.6
        if "category_ids" in kwargs:
            category_labels = kwargs["category_ids"].copy()
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            # print('gt mask', gt_mask)
            # print('pred mask', pred_mask.shape)
            # print('=' * 100)
            category_label = category_labels[batch_idx]
            weight = calculate_weights(gt_mask, category_label, lambda_weight=lambda_class_weight)
            
            # gt_dim = gt_mask.shape[0]
            # pred_dim = pred_mask.shape[0]
            # if gt_dim != pred_dim:
            #     pred_mask = pred_mask[:gt_dim, ...]
                
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, weight, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            variation_loss += total_variation_loss(pred_mask.sigmoid())
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        variation_loss = lambda_tv * variation_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        if "category_ids" in kwargs:
            category_labels = kwargs["category_ids"]
            classification_losses = []

            for i in range(len(pred_embeddings)):
                emb = pred_embeddings[i]
                label = category_labels[i]

                # emb_mean = emb.mean(dim=0, keepdim=True)  # [1, hidden_size]
                logits = self.model.classification_head(emb)  # [1, num_classes]
                # print('logits', logits.shape)
                # print('label', label.shape)
                # print('='* 100)
                # logit_dim = logits.shape[0]
                # label_dim = label.shape[0]
                # if logit_dim != label_dim:
                #     logits = logits[:label_dim,...]
                loss_cls = F.cross_entropy(logits, label)
                classification_losses.append(loss_cls)
            if classification_losses:
                classification_loss = sum(classification_losses) / len(
                    classification_losses
                )
            else:
                classification_loss = 0.0
        else:
            classification_loss = 0.0

        loss = ce_loss + mask_loss + variation_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "classification_loss": classification_loss,
            "variation_loss": variation_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=2048,
        tokenizer=None,
        local_rank=0,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx

            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings, early_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []

            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,  # [# of tokens, 1, 256]
                    dense_embeddings,  # [# of tokens, 256, 64, 64]
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=(
                        image_embeddings[i].unsqueeze(0),
                        early_embeddings[i].unsqueeze(0),
                    ),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )  # [# of tokens, 1, 256, 256]

                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )

                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
