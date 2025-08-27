import glob
import json
import os
import random
import cv2
import numpy as np
import transformers

import torch
import torch.nn.functional as F

from unicodedata import category
from pycocotools import mask

from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib


from .utils import (
    MR_SINGLE_ANSWER_LIST,
    MR_MULTI_ANSWER_LIST,
    ANSWER_LIST,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    EXPLANATORY_QUESTION_LIST,
    LONG_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    EXPAND_LONG_QUESTION_LIST,
)


class MultiPartReasonSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        train_json_file,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        use_expand_question_list=False,
    ):
        self.samples_per_epoch = samples_per_epoch

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.single_answer_list = MR_SINGLE_ANSWER_LIST
        self.multi_answer_list = MR_MULTI_ANSWER_LIST

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.num_classes_per_sample = num_classes_per_sample
        self.train_json_file = train_json_file

        if use_expand_question_list:
            self.long_question_list.extend(EXPAND_LONG_QUESTION_LIST)

        json_file_name = os.path.join(
            self.base_image_dir, f"{self.train_json_file}.json"
        )

        with open(json_file_name, "r") as f:
            reason_file = json.load(f)

        self.reason_part_seg_data = reason_file["data"]
        print("# of part_reason_seg samples: ", len(reason_file))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.reason_part_seg_data) - 1)
        image_info = self.reason_part_seg_data[idx]

        image_path = image_info["img_origin"]

        question = image_info["questions"]
        gt_answer = image_info["answers"]
        text_answers = image_info["text_answers"]
        
        image_path = image_path.replace('/app/iccv2025/PathVR', '..')
        # print('image path', image_path)
        # print('question', question)
        # print('gt answer', gt_answer)
        # print('text answer', text_answers)
        # print('=' * 100 )
        
        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocess images for clip
        image_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ][0]

        images = self.transform.apply_image(images)
        resize = images.shape[:2]
        masks = []

        # NOTE: maximum num_classes_per_sample questions per sample
        if len(question) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(question))),
                size=self.num_classes_per_sample,
                replace=False,
            )
        else:
            sampled_inds = list(range(len(question)))

        sampled_sents = np.vectorize(question.__getitem__)(sampled_inds).tolist()

        sampled_answers = gt_answer
        sampled_masks = masks
        sampled_text_answers = text_answers

        questions = []
        answers = []
        category_ids = []

        if len(question) != 0:
            for text, answer_list, text_answer in zip(
                sampled_sents, sampled_answers, sampled_text_answers
            ):
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text))

                for answer in answer_list:
                    rle = answer["segmentation"]
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)
                    masks.append(m)

                    _text_answer = text_answer
                    answers.append(_text_answer)
                    category_ids.append(answer["category_id"])

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        # print('conv', conv)
        # print('=' * 100)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            # print('image path', image_path)
            # print('question len', len(questions))
            # print('conv role', conv.roles[1], i)
            # print('answers', answers)
            # print('=' * 100)
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())

        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        category_ids = torch.tensor(category_ids)

        return (
            image_path,
            images,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            category_ids,
        )
