import numpy as np
import torch

from .alpaca_gpt4_dataset import AlpacaGPT4Dataset  # noqa: F401
from .aokvqa_dataset import AOKVQADataset  # noqa: F401
from .cc_sbu_align_dataset import CcSbuAlignDataset  # noqa: F401
from .clevr_dataset import CLEVRDataset  # noqa: F401
from .coco_caption_dataset import COCOCaptionDataset  # noqa: F401
from .dial_dataset import DialDataset  # noqa: F401
from .dolly_dataset import DollyDataset  # noqa: F401
from .gqa_dataset import GQADataset  # noqa: F401
from .llava_dataset import LlavaDataset  # noqa: F401
from .nlvr_dataset import NLVRv1Dataset, NLVRv2Dataset  # noqa: F401
from .ocr_vqa_dataset import OCRVQADataset  # noqa: F401
from .snli_ve_datasets import SNLIVEDataset  # noqa: F401
from .text_ocr_dataset import TextOCRDataset  # noqa: F401
from .vqa_dataset import ConcatDataset, VQADataset  # noqa: F401
from .baize_dataset import BaiZeDataset  # noqa: F401


def build_dataset(dataset_config, **kwargs):
    if isinstance(dataset_config, list):
        datasets = [build_dataset(cfg, **kwargs) for cfg in dataset_config]
        return ConcatDataset(datasets)
    dataset_type = dataset_config.pop("type")
    sample = dataset_config.pop("sample", -1)
    if dataset_type == "llava":
        dataset = LlavaDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "vqa":
        dataset = VQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "minigpt4":
        dataset = CcSbuAlignDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "llava_dial":
        dataset = DialDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "coco_dial":
        dataset = DialDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "aokvqa":
        dataset = AOKVQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "okvqa":
        dataset = VQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "text_ocr":
        dataset = TextOCRDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "ocr_vqa":
        dataset = OCRVQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "coco_caption":
        dataset = COCOCaptionDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "gqa":
        dataset = GQADataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "clevr":
        dataset = CLEVRDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "nlvrv1":
        dataset = NLVRv1Dataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "nlvrv2":
        dataset = NLVRv2Dataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "snlive":
        dataset = SNLIVEDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "dolly":
        dataset = DollyDataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "alpaca_gpt4":
        dataset = AlpacaGPT4Dataset(
            **dataset_config,
            **kwargs,
        )
    elif dataset_type == "baize":
        dataset = BaiZeDataset(
            **dataset_config,
            **kwargs,
        )
    else:
        raise NotImplementedError

    if sample > 0:
        random_indices = np.random.choice(len(dataset), min(sample, len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.collater = dataset.collater
        return subsample_dataset
    else:
        return dataset
