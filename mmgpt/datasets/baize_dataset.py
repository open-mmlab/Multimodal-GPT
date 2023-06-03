import json

from mmgpt.datasets.dolly_dataset import DollyDataset


TEMPLATE = {
    "description": "Template used by Alpaca-LoRA.",
    "prompt_choice": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Input:\n{options}\n\n### Response:\n",
    "prompt_qa": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n### Response:\n",
    "prompt_dial": "\n\n### Instruction:\n{question}\n\n### Response:\n",
    "response_split": "### Response:",
}

class LangDialPrompter:
    def __call__(self, question, options=None):
        if options:
            options = ", ".join(options)
            res = TEMPLATE["prompt_choice"].format(image="<image>", question=question, options=options)
        else:
            res = TEMPLATE["prompt_dial"].format(question=question)
        return res

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()

class BaiZeDataset(DollyDataset):
    """
    ```json
    [
        {
            "instruction": "Identify the odd one out.",
            "input": "Twitter, Instagram, Telegram",
            "output": "The odd one out is Telegram. Twitter and Instagram are social media platforms mainly for sharing information, images and videos while Telegram is a cloud-based instant messaging and voice-over-IP service."
        },
    ]
    """
    def __init__(self, *args, **kwargs):
        super(BaiZeDataset, self).__init__(*args, **kwargs)
        self.prompter = LangDialPrompter()

    def load_annotation(self, ann_path):
        self.annotation = json.load(open(ann_path, "r"))

    def process_text(self, anns):
        # TODO remove this
        begin_string = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        convs = anns['input'].split("[|Human|] ")
        conv_list = []
        for conv_id, one_conv in enumerate(convs[1:-1]):
            question, answer = one_conv.split("[|AI|] ")
            question = question.replace("\n", "")
            answer = answer.replace("\n", "")
            instruction = self.prompter(question)
            if conv_id == 0:
                single_conv = dict(instruction=begin_string + instruction, answer=answer)
            else:
                single_conv = dict(instruction=instruction, answer=answer)
            conv_list.append(single_conv)
        return conv_list
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        text_list = self.process_text(ann)
        res_list = []
        for text in text_list:
            single_res = self.tokenize(text)
            single_res["instruction"] = text["instruction"]
            single_res["answer"] = text["answer"]
            res_list.append(single_res)

        input_ids = []
        attention_mask = []
        labels = []
        instruction = []
        answer = []
        for res in res_list:
            input_ids.extend(res["input_ids"])
            attention_mask.extend(res["attention_mask"])
            labels.extend(res["labels"])
            instruction.append(res["instruction"])
            answer.append(res["answer"])

        res = dict(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, instruction=instruction, answer=answer
        )
        return res
