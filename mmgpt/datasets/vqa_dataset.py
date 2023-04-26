"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import LlamaTokenizer

TEMPLATE = {
    "description": "Template used by Alpaca-LoRA.",
    # "prompt_choice": "Below is a multiple choice question about an image, along with answer options. Please choose the correct answer from these options.\n\n### Image:\n{image}\n\n### Question:\n{question}\n\n### Input:\n{options}\n\n### Answer:\n",
    # "prompt_qa": "Below is a question about an image. Write a response to answer the question.\n\n### Image:\n{image}\n\n### Question:\n{question}\n\n### Answer:\n",
    "prompt_choice": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Input:\n{options}\n\n### Response:\n",
    "prompt_qa": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Response:\n",
    "response_split": "### Response:",
}


class VQAPrompter:
    def __call__(self, question, options=None):
        if options:
            options = ", ".join(options)
            res = TEMPLATE["prompt_choice"].format(image="<image>", question=question, options=options)
        else:
            res = TEMPLATE["prompt_qa"].format(image="<image>", question=question)
        return res

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()


class VQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        vis_processor=None,
        vis_root=None,
        ann_paths=[],
        add_eos=True,
        ignore_instruction=True,
        sample_image=False,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer: LlamaTokenizer = tokenizer
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.sample_image = sample_image
        if self.sample_image:
            print("randomly sample one annotation for each image")
            self.annotation = self.parse_annotation(self.annotation)

        self.vis_processor = vis_processor

        self._add_instance_ids()
        self.option_prob = 0.5
        self.prompter = VQAPrompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

    def parse_annotation(self, annotation):
        image_list = defaultdict(list)
        for ann in annotation:
            image_list[ann["image"]].append(ann)
        # image_name_list = list(image_list.keys())
        annotation = []
        for ann_list in image_list.values():
            annotation.append(random.choice(ann_list))
        return annotation

    def __len__(self):
        return len(self.annotation)

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        return image

    def process_text(self, ann):
        question = ann["question"]

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        # create instruction
        true_answer = answers[np.argmax(weights)]
        is_option = random.random() < self.option_prob and len(answers) > 1
        if is_option:
            instruction = self.prompter(question, answers)
        else:
            instruction = self.prompter(question)

        return dict(instruction=instruction, answer=true_answer)

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"] + text["answer"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )

        # manually add eos token
        if res["input_ids"][-1] != self.tokenizer.eos_token_id and len(res["input_ids"]) < 512 and self.add_eos:
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [-100] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = self.process_image(ann)
        text = self.process_text(ann)
        res = self.tokenize(text)
        res.update(image=image)
        res.update(text)
        return res

    def collater(self, samples):
        image_list, question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_label_length = max(len(l) for l in labels_list)
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        labels = padded_samples["labels"]
        media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
