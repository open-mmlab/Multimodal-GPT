from .vqa_dataset import VQADataset

TEMPLATE = {
    "description": "Template used by Alpaca-LoRA.",
    # "prompt_choice": "Below is a multiple choice question about an image, along with answer options. Please choose the correct answer from these options.\n\n### Image:\n{image}\n\n### Question:\n{question}\n\n### Options:\n{options}\n\n### Answer:\n",
    # "prompt_qa": "Below is a question about an image. Write a response to answer the question.\n\n### Image:\n{image}\n\n### Question:\n{question}\n\n### Answer:\n",
    "prompt_choice": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Input:\n{options}\n\n### Response:\n",
    "prompt_qa": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Response:\n",
    "prompt_dial": "\n\n### Instruction:\n{question}\n\n### Response:\n",
    "response_split": "### Response:",
}


class DialPrompter:
    def __call__(self, question, options=None):
        if options:
            options = ", ".join(options)
            res = TEMPLATE["prompt_choice"].format(image="<image>", question=question, options=options)
        else:
            res = TEMPLATE["prompt_dial"].format(question=question)
        return res

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()


class DialDataset(VQADataset):
    def __init__(self, *args, **kwargs):
        super(DialDataset, self).__init__(*args, **kwargs)
        self.prompter = DialPrompter()

    def _add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

    def process_text(self, anns):
        # TODO remove this
        begin_string = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}".format(
            image="<image>"
        )
        num_convs = len(anns["conversations"]) // 2
        conv_list = []
        for conv_id in range(num_convs):
            question = anns["conversations"][conv_id]["value"]
            # remove '<image>' tag and '\n'
            question = question.replace("<image>", "").replace("\n", "")
            answer = anns["conversations"][conv_id + 1]["value"]
            instruction = self.prompter(question)
            if conv_id == 0:
                single_conv = dict(instruction=begin_string + instruction, answer=answer)
            else:
                single_conv = dict(instruction=instruction, answer=answer)
            conv_list.append(single_conv)
        return conv_list

    def __getitem__(self, index):
        ann = self.annotation[index]
        image = self.process_image(ann)
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
            instruction.extend(res["instruction"])
            answer.extend(res["answer"])

        res = dict(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, instruction=instruction, answer=answer
        )
        res.update(image=image)
        return res
