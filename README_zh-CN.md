# 🤖 Multi-modal GPT

使用视觉和语言指令训练一个多模态聊天机器人！

基于开源多模态模型 [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)，我们使用公开数据集创建了各种**视觉指令**数据，包括视觉问答、图像字幕、视觉推理、文本 OCR 和视觉对话。此外，我们还使用仅包含**语言指令**数据的语言模型组件进行了训练。

视觉和语言指令的**联合训练**有效提高了模型的性能！更多细节请参阅我们的[技术报告](https://arxiv.org/abs/2305.04790)。

欢迎加入我们！

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## 特性

- 支持各种视觉和语言指令数据
- 使用 LoRA 进行参数高效微调
- 同时调整视觉和语言，相互补充

## 安装

在一个已有环境中安装依赖包，运行以下指令

```bash
git clone https://github.com/open-mmlab/Multimodal-GPT.git
cd Multimodal-GPT
pip install -r requirements.txt
pip install -v -e .
```

或者创建一个新的 conda 环境

```bash
conda env create -f environment.yml
```

## Demo

1. 下载预训练权重

    使用[这个脚本](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)把 LLaMA 权重转换成 HuggingFace 格式。

    从 [openflamingo/OpenFlamingo-9B](https://huggingface.co/openflamingo/OpenFlamingo-9B) 下载 OpenFlamingo 预训练模型。

    从[这个链接](https://download.openmmlab.com/mmgpt/v0/mmgpt-lora-v0-release.pt) 下载我们的 LoRA 权重。

    然后把所有模型权重放到 `checkpoints` 文件夹下，目录结构如下：

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

2. 启动 gradio demo

    ```bash
    python app.py
    ```

## 示例

### 菜单：
![image4](https://user-images.githubusercontent.com/12907710/234554562-8f3be88f-d563-47ba-97d9-ade8d47c46b0.png)

### 旅行计划：
![image3](https://user-images.githubusercontent.com/12907710/234523464-80c4e3f0-f99f-4498-96ef-dc43ef89c64b.png)

### 电影：
![image2](https://user-images.githubusercontent.com/12907710/234523468-e11905a6-491f-4b87-934f-90da7d14d1c3.png)

### 名人：
![image](https://user-images.githubusercontent.com/12907710/234523475-fd91f979-a344-4228-813f-6b55a1bc250f.png)


## 微调 Fine-tuning

### 准备数据集

1. [A-OKVQA](https://allenai.org/project/a-okvqa/home)

    从[这个链接](https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz)下载标注，解压到 `data/aokvqa/annotations` 路径下。

    同时还需要 coco 数据集的图像，可以从[这里](https://cocodataset.org/#home)下载。

2. [COCO Caption](https://cs.stanford.edu/people/karpathy/deepimagesent/)

    从[这个链接](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip)，解压到 `data/coco` 路径下。

    同时还需要 coco 数据集的图像，可以从[这里](https://cocodataset.org/#home)下载。

3. [OCR VQA](https://ocr-vqa.github.io/)

    从 [这个链接](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) 下载数据集，放到 `data/OCR_VQA/` 路径下。

4. [LlaVA](https://llava-vl.github.io/)

    从 [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) 下载数据集，放到 `data/llava/` 路径下。

    同时还需要 coco 数据集的图像，可以从[这里](https://cocodataset.org/#home)下载。

5. [Mini-GPT4](https://minigpt-4.github.io/)

    从 [Vision-CAIR/cc_sbu_align](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) 下载数据集，放到 `data/cc_sbu_align/` 路径下。

6. [Dolly 15k](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)

    从 [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 下载数据集，放到 `data/dolly/databricks-dolly-15k.jsonl` 路径下。

7. [Alpaca GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

    从[这个链接](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json) 下载数据集，放到 `data/alpaca_gpt4/alpaca_gpt4_data.json` 路径下。

你也可以在 [configs/dataset_config.py](configs/dataset_config.py) 文件中自定义数据集路径。


## 开启训练

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
  --report_to_wandb
```


## 致谢

- [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)
- [LAVIS](https://github.com/salesforce/LAVIS)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main)
- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

如果你觉得我们的项目对你的研究和应用有帮助，请用以下 BibTeX 进行引用

```bibtex
@misc{gong2023multimodalgpt,
      title={MultiModal-GPT: A Vision and Language Model for Dialogue with Humans}, 
      author={Tao Gong and Chengqi Lyu and Shilong Zhang and Yudong Wang and Miao Zheng and Qian Zhao and Kuikun Liu and Wenwei Zhang and Ping Luo and Kai Chen},
      year={2023},
      eprint={2305.04790},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
