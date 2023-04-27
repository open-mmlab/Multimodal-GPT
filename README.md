# 🤖 Multi-modal GPT

Train a multi-modal chatbot with visual and language instructions! 

Based on the open-source multi-modal model [OpenFlamingo](https://github.com/mlfoundations/open_flamingo), we create various **visual instruction** data with open datasets, including VQA, Image Captioning, Visual Reasoning, Text OCR, and Visual Dialogue. Additionally, we also train the language model component of OpenFlamingo using only **language-only instruction** data.

The **joint training** of visual and language instructions effectively improves the performance of the model!

# Features

- Support various vision and language instruction data
- Parameter efficient fine-tuning with LoRA
- Tuning vision and language at the same time, complement each other

# Installaion

To install the package in an existing environment, run

```bash
git clone https://github.com/open-mmlab/Multimodal-GPT.git
pip install -r requirements.txt
pip install -e. -v
```

or create a new conda environment

```bash
conda env create -f environment.yml
```


# Demo

1. Download the pre-trained weights.

    Use [this script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) for converting LLaMA weights to HuggingFace format.

    Download the OpenFlamingo pre-trained model from [openflamingo/OpenFlamingo-9B](https://huggingface.co/openflamingo/OpenFlamingo-9B)

    Download our LoRA Weight from [here](https://download.openmmlab.com/mmgpt/v0/mmgpt-lora-v0-release.pt)

    Then place these models in checkpoints folders like this:

    ```
    checkpoints
    ├── llama-7b_hf
    │   ├── config.json
    │   ├── pytorch_model-00001-of-00002.bin
    │   ├── ......
    │   └── tokenizer.model
    ├── OpenFlamingo-9B
    │   └──checkpoint.pt
    ├──mmgpt-lora-v0-release.pt

2. launch the gradio demo

    ```bash
    python chat_gradio_demo.py
    ```

# Examples

### Recipe:
![image4](https://user-images.githubusercontent.com/12907710/234554562-8f3be88f-d563-47ba-97d9-ade8d47c46b0.png)

### Travel plan:
![image3](https://user-images.githubusercontent.com/12907710/234523464-80c4e3f0-f99f-4498-96ef-dc43ef89c64b.png)
### Movie:
![image2](https://user-images.githubusercontent.com/12907710/234523468-e11905a6-491f-4b87-934f-90da7d14d1c3.png)
### Famous person:
![image](https://user-images.githubusercontent.com/12907710/234523475-fd91f979-a344-4228-813f-6b55a1bc250f.png)


# Fine-tuning

## Prepare datasets

1. [A-OKVQA](https://allenai.org/project/a-okvqa/home)

    Download annotation from [this link](https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz) and unzip to `data/aokvqa/annotations`

    It also requires images from coco dataset which can be downloaded from [here](https://cocodataset.org/#home). 

2. [COCO Caption](https://cs.stanford.edu/people/karpathy/deepimagesent/)

    Download from [this link](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip) and unzip to `data/coco`

    It also requires images from coco dataset which can be downloaded from [here](https://cocodataset.org/#home).

3. [OCR VQA](https://ocr-vqa.github.io/)

    Download from [this link](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) and place in `data/OCR_VQA/`

4. [LlaVA](https://llava-vl.github.io/)

    Download from [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and place in `data/llava/`

    It also requires images from coco dataset which can be downloaded from [here](https://cocodataset.org/#home).

5. [Mini-GPT4](https://minigpt-4.github.io/)

    Download from [Vision-CAIR/cc_sbu_align](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) and place in `data/cc_sbu_align/`

6. [Dolly 15k](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)

    Download from [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and place it in `data/dolly/databricks-dolly-15k.jsonl`

7. [Alpaca GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

    Download it from [this link](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json) and place it in `data/alpaca_gpt4/alpaca_gpt4_data.json`

You can also customize the data path in the [configs/dataset_config.py](configs/dataset_config.py).


## Start training

```bash
torchrun --nproc_per_node=8 mmgpt/train/instruction_finetune.py \
--lm_path checkpoints/llama-7b_hf \
--tokenizer_path checkpoints/llama-7b_hf \
--pretrained_path checkpoints/OpenFlamingo-9B/checkpoint.pt \
--run_name train-my-gpt4 \
--learning_rate 1e-5 \
--lr_scheduler cosine \
--batch_size 1 \ 
--tuning_config configs/lora_config.py \
--dataset_config configs/dataset_config.py \
--report_to_wandb \
```


# Acknowledgements

- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
- [LAVIS](https://github.com/salesforce/LAVIS)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main)
- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
