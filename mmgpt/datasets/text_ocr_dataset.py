import json
import os
import random

import numpy as np
from PIL import Image
from transformers import LlamaTokenizer

from .vqa_dataset import VQADataset, VQAPrompter


class TextOCRDataset(VQADataset):
    def __init__(
        self, tokenizer, vis_processor=None, vis_root=None, ann_paths=[], add_eos=True, ignore_instruction=True
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
            self.annotation.extend(json.load(open(ann_path, "r"))["data"])

        self.vis_processor = vis_processor

        self._add_instance_ids()
        self.option_prob = 0.5
        self.prompter = VQAPrompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["image_id"] + ".jpg")
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        return image

    def process_text(self, ann):
        question = ann["question"]

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

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
