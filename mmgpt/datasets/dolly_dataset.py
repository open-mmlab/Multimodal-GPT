import copy
import json

import numpy as np
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

TEMPLATE = {
    "description": "Template used by LLM.",
    "prompt_no_input_format": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "prompt_with_input_format": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "response_split": "### Response:",
}


class LMPrompter:
    def __call__(self, instruction, input=None):
        if input is None or len(input) == 0:
            return TEMPLATE["prompt_no_input_format"].format(instruction=instruction)
        else:
            return TEMPLATE["prompt_with_input_format"].format(instruction=instruction, input=input)

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()


class DollyDataset(Dataset):
    """Each line of the annotation file is a json object with the following fields:

    {
        "instruction": "What is a dispersive prism?",
        "context": "In optics, a dispersive prism is an optical prism that is used to disperse light, that is, to separate light into its spectral components (the colors of the rainbow). Different wavelengths (colors) of light will be deflected by the prism at different angles.[1] This is a result of the prism material's index of refraction varying with wavelength (dispersion). Generally, longer wavelengths (red) undergo a smaller deviation than shorter wavelengths (blue). The dispersion of white light into colors by a prism led Sir Isaac Newton to conclude that white light consisted of a mixture of different colors.",
        "response": "A dispersive prism is an optical prism that disperses the light's different wavelengths at different angles. When white light is shined through a dispersive prism it will separate into the different colors of the rainbow.",
        "category": "summarization"
    }

    """

    def __init__(self, tokenizer, ann_path: str, add_eos=True, ignore_instruction=True, **kwargs):
        """
        ann_path (string): directory to store the annotation file
        """
        assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer: LlamaTokenizer = tokenizer

        self.annotation = []
        self.prompter = LMPrompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.load_annotation(ann_path)

    def load_annotation(self, ann_path):
        self.annotation = []
        for line in open(ann_path, "r").readlines():
            self.annotation.append(json.loads(line))

    def __len__(self):
        return len(self.annotation)

    def process_text(self, ann):
        instruction = ann["instruction"]
        context = ann["context"]
        response = ann["response"]
        instruction = self.prompter(instruction=instruction, input=context)
        return dict(instruction=instruction, answer=response)

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
        text = self.process_text(ann)
        res = self.tokenize(text)
        res.update(text)
        return res

    def collater(self, samples):
        question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], []

        for sample in samples:
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
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        return {
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }


def build_dolly_dataset(
    tokenizer,
    ann_path="data/dolly/databricks-dolly-15k.jsonl",
    **kwargs,
):
    return DollyDataset(
        tokenizer=tokenizer,
        ann_path=ann_path,
        **kwargs,
    )
