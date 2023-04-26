import random

from .vqa_dataset import VQADataset

REASON_QUESTIONS = [
    "Why?",
    "Why is this?",
    "And why?",
    "What is the reason?",
    "And can you tell me why?",
    "Can you tell me why?",
    "Can you tell me the reason?",
]


class AOKVQADataset(VQADataset):
    def __init__(self, tokenizer, vis_processor, vis_root, ann_paths, **kwargs):
        super().__init__(tokenizer, vis_processor, vis_root, ann_paths, **kwargs)

    def process_text(self, ann):
        question = ann["question"]
        question = question + " " + random.choice(REASON_QUESTIONS)

        choices = ann["choices"]
        true_answer = choices[ann["correct_choice_idx"]]
        answer = "The answer is " + true_answer + ". Because " + " ".join(ann["rationales"])

        is_option = random.random() < self.option_prob and len(choices) > 1
        if is_option:
            instruction = self.prompter(question, choices)
        else:
            instruction = self.prompter(question)

        instruction = self.prompter(question)
        return dict(instruction=instruction, answer=answer)


def build_aokvqa_dataset(
    tokenizer,
    vis_processor,
    vis_root="data/coco/images",
    ann_paths=["data/aokvqa/annotations/aokvqa_v1p0_train.json"],
    sample_image=False,
):
    return AOKVQADataset(
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        vis_root=vis_root,
        ann_paths=ann_paths,
        sample_image=sample_image,
    )
