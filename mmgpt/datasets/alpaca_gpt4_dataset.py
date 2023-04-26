import json

from mmgpt.datasets.dolly_dataset import DollyDataset


class AlpacaGPT4Dataset(DollyDataset):
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

    def load_annotation(self, ann_path):
        self.annotation = json.load(open(ann_path, "r"))

    def process_text(self, ann):
        instruction = ann["instruction"]
        input = ann["input"]
        output = ann["output"]
        instruction = self.prompter(instruction=instruction, input=input)
        return dict(instruction=instruction, answer=output)
