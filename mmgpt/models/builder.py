from .open_flamingo import create_model_and_transforms as create_open_flamingo_model_and_transforms


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
