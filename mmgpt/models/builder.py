from .open_flamingo import create_model_and_transforms as create_open_flamingo_model_and_transforms
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM

def create_model_and_transforms(
    model_name: str,
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    tuning_config,
    pretrained_model_path,
    **kwargs,
):
    if model_name == "open_flamingo":
        return create_open_flamingo_model_and_transforms(
            clip_vision_encoder_path=clip_vision_encoder_path,
            clip_vision_encoder_pretrained=clip_vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            tuning_config=tuning_config,
            pretrained_model_path=pretrained_model_path,
            **kwargs,
        )
    # TODO: support BLIP2
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# only for debugging
def create_toy_model_and_transforms(
    model_name: str,
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    tuning_config,
    pretrained_model_path,
    **kwargs,
):
    print("init toy vision encoder")
    import torchvision

    image_processor = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )
    print("init tokenizer")
    text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    class ToyModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.input_embeddings = nn.Embedding(38000, 512)
            self.layer = nn.Linear(512, 512)
            self.config = {"hidden_size": 512}

        def forward(self, lang_x, **kwargs):
            x = self.input_embeddings(lang_x)
            x = self.layer(x)
            loss = x.sum()

            return (loss,)

    model = ToyModel()

    return model, image_processor, text_tokenizer
