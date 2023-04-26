"""Modified from https://github.com/mlfoundations/open_flamingo"""
import open_clip
import torch
import torch.nn as nn
from bigmodelvis import Visualization
from peft import LoraConfig, get_peft_model
from transformers import LlamaForCausalLM, LlamaTokenizer

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance


def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    decoder_layers_attr_name: str = None,
    pretrained_model_path: str = None,
    tuning_config=None,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    print("init clip vision encoder")
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True
    print("init tokenizer")
    text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2

    print("init llama")
    lang_encoder = LlamaForCausalLM.from_pretrained(lang_encoder_path)
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"]["width"],
        cross_attn_every_n_layers=4,
        **flamingo_kwargs,
    )

    if pretrained_model_path is not None:
        print(f"loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    if tuning_config is not None:
        model = prepare_model_for_tuning(model, tuning_config)
    else:
        raise ValueError("tuning_config must be provided")

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
}


def prepare_model_for_tuning(model: nn.Module, config):
    if config.lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",  # won't use bias currently
            modules_to_save=[],  # TODO: might be helpful if save partial model
            task_type="VL",
        )
        model.lang_encoder = get_peft_model(model.lang_encoder, peft_config=lora_config)

    # manually unfreeze modules, we use a `substring` fashion mathcing
    for name, param in model.named_parameters():
        if any(substr in name for substr in config.unfrozen):
            param.requires_grad = True

    if config.vis and is_rank0():
        Visualization(model).structure_graph()
    return model


# temporary workaround, should use a common utils in the future
def is_rank0():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
