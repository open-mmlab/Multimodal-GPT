import os

import gradio as gr
import torch
from PIL import Image

from mmgpt.models.builder import create_model_and_transforms

TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
response_split = "### Response:"


class Inferencer:

    def __init__(self, finetune_path, llama_path, open_flamingo_path):
        ckpt = torch.load(finetune_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # remove the "module." prefix
            state_dict = {
                k[7:]: v
                for k, v in state_dict.items() if k.startswith("module.")
            }
        else:
            state_dict = ckpt
        tuning_config = ckpt.get("tuning_config")
        if tuning_config is None:
            print("tuning_config not found in checkpoint")
        else:
            print("tuning_config found in checkpoint: ", tuning_config)
        model, image_processor, tokenizer = create_model_and_transforms(
            model_name="open_flamingo",
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            pretrained_model_path=open_flamingo_path,
            tuning_config=tuning_config,
        )
        model.load_state_dict(state_dict, strict=False)
        model.half()
        model = model.to("cuda")
        model.eval()
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, prompt, imgpaths, max_new_token, num_beams, temperature,
                 top_k, top_p, do_sample):
        if len(imgpaths) > 1:
            raise gr.Error(
                "Current only support one image, please clear gallery and upload one image"
            )
        lang_x = self.tokenizer([prompt], return_tensors="pt")
        if len(imgpaths) == 0 or imgpaths is None:
            for layer in self.model.lang_encoder._get_decoder_layers():
                layer.condition_only_lang_x(True)
            output_ids = self.model.lang_encoder.generate(
                input_ids=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )[0]
            for layer in self.model.lang_encoder._get_decoder_layers():
                layer.condition_only_lang_x(False)
        else:
            images = (Image.open(fp) for fp in imgpaths)
            vision_x = [self.image_processor(im).unsqueeze(0) for im in images]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0).half()

            output_ids = self.model.generate(
                vision_x=vision_x.cuda(),
                lang_x=lang_x["input_ids"].cuda(),
                attention_mask=lang_x["attention_mask"].cuda(),
                max_new_tokens=max_new_token,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )[0]
        generated_text = self.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        # print(generated_text)
        result = generated_text.split(response_split)[-1].strip()
        return result


class PromptGenerator:

    def __init__(
        self,
        prompt_template=TEMPLATE,
        ai_prefix="Response",
        user_prefix="Instruction",
        sep: str = "\n\n### ",
        buffer_size=0,
    ):
        self.all_history = list()
        self.ai_prefix = ai_prefix
        self.user_prefix = user_prefix
        self.buffer_size = buffer_size
        self.prompt_template = prompt_template
        self.sep = sep

    def add_message(self, role, message):
        self.all_history.append([role, message])

    def get_images(self):
        img_list = list()
        if self.buffer_size > 0:
            all_history = self.all_history[-2 * (self.buffer_size + 1):]
        elif self.buffer_size == 0:
            all_history = self.all_history[-2:]
        else:
            all_history = self.all_history[:]
        for his in all_history:
            if type(his[-1]) == tuple:
                img_list.append(his[-1][-1])
        return img_list

    def get_prompt(self):
        format_dict = dict()
        if "{user_prefix}" in self.prompt_template:
            format_dict["user_prefix"] = self.user_prefix
        if "{ai_prefix}" in self.prompt_template:
            format_dict["ai_prefix"] = self.ai_prefix
        prompt_template = self.prompt_template.format(**format_dict)
        ret = prompt_template
        if self.buffer_size > 0:
            all_history = self.all_history[-2 * (self.buffer_size + 1):]
        elif self.buffer_size == 0:
            all_history = self.all_history[-2:]
        else:
            all_history = self.all_history[:]
        context = []
        have_image = False
        for role, message in all_history[::-1]:
            if message:
                if type(message) is tuple and message[
                        1] is not None and not have_image:
                    message, _ = message
                    context.append(self.sep + "Image:\n<image>" + self.sep +
                                   role + ":\n" + message)
                else:
                    context.append(self.sep + role + ":\n" + message)
            else:
                context.append(self.sep + role + ":\n")

        ret += "".join(context[::-1])
        return ret


def to_gradio_chatbot(prompt_generator):
    ret = []
    for i, (role, msg) in enumerate(prompt_generator.all_history):
        if i % 2 == 0:
            if type(msg) is tuple:
                import base64
                from io import BytesIO

                msg, image = msg
                if type(image) is str:
                    from PIL import Image

                    image = Image.open(image)
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 800, 400
                shortest_edge = int(
                    min(max_len / aspect_ratio, min_len, min_hw))
                longest_edge = int(shortest_edge * aspect_ratio)
                H, W = image.size
                if H > W:
                    H, W = longest_edge, shortest_edge
                else:
                    H, W = shortest_edge, longest_edge
                image = image.resize((H, W))
                # image = image.resize((224, 224))
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                msg = msg + img_str
            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret


def bot(
    text,
    image,
    state,
    prompt,
    ai_prefix,
    user_prefix,
    seperator,
    history_buffer,
    max_new_token,
    num_beams,
    temperature,
    top_k,
    top_p,
    do_sample,
):
    state.prompt_template = prompt
    state.ai_prefix = ai_prefix
    state.user_prefix = user_prefix
    state.sep = seperator
    state.buffer_size = history_buffer
    if image:
        state.add_message(user_prefix, (text, image))
    else:
        state.add_message(user_prefix, text)
    state.add_message(ai_prefix, None)
    inputs = state.get_prompt()
    image_paths = state.get_images()[-1:]

    inference_results = inferencer(inputs, image_paths, max_new_token,
                                   num_beams, temperature, top_k, top_p,
                                   do_sample)
    state.all_history[-1][-1] = inference_results
    memory_allocated = str(round(torch.cuda.memory_allocated() / 1024**3,
                                 2)) + 'GB'
    return state, to_gradio_chatbot(state), "", None, inputs, memory_allocated


def clear(state):
    state.all_history = []
    return state, to_gradio_chatbot(state), "", None, ""


title_markdown = ("""
    # 🤖 Multi-modal GPT
    [[Project]](https://github.com/open-mmlab/Multimodal-GPT.git)""")


def build_conversation_demo():
    with gr.Blocks(title="Multi-modal GPT") as demo:
        gr.Markdown(title_markdown)

        state = gr.State(PromptGenerator())
        with gr.Row():
            with gr.Column(scale=3):
                memory_allocated = gr.Textbox(
                    value=init_memory, label="Memory")
                imagebox = gr.Image(type="filepath")
                # TODO config parameters
                with gr.Accordion(
                        "Parameters",
                        open=True,
                ):
                    max_new_token_bar = gr.Slider(
                        0, 1024, 512, label="max_new_token", step=1)
                    num_beams_bar = gr.Slider(
                        0.0, 10, 3, label="num_beams", step=1)
                    temperature_bar = gr.Slider(
                        0.0, 1.0, 1.0, label="temperature", step=0.01)
                    topk_bar = gr.Slider(0, 100, 20, label="top_k", step=1)
                    topp_bar = gr.Slider(0, 1.0, 1.0, label="top_p", step=0.01)
                    do_sample = gr.Checkbox(True, label="do_sample")
                with gr.Accordion(
                        "Prompt",
                        open=False,
                ):
                    with gr.Row():
                        ai_prefix = gr.Text("Response", label="AI Prefix")
                        user_prefix = gr.Text(
                            "Instruction", label="User Prefix")
                        seperator = gr.Text("\n\n### ", label="Seperator")
                    history_buffer = gr.Slider(
                        -1, 10, -1, label="History buffer", step=1)
                    prompt = gr.Text(TEMPLATE, label="Prompt")
                    model_inputs = gr.Textbox(label="Actual inputs for Model")

            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column():
                        chatbot = gr.Chatbot(elem_id="chatbot").style(
                            height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                        ).style(container=False)
                        submit_btn = gr.Button(value="Submit")
                        clear_btn = gr.Button(value="🗑️  Clear history")
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        gr.Examples(
            examples=[
                [
                    f"{cur_dir}/docs/images/demo_image.jpg",
                    "What is in this image?"
                ],
            ],
            inputs=[imagebox, textbox],
        )
        textbox.submit(
            bot,
            [
                textbox,
                imagebox,
                state,
                prompt,
                ai_prefix,
                user_prefix,
                seperator,
                history_buffer,
                max_new_token_bar,
                num_beams_bar,
                temperature_bar,
                topk_bar,
                topp_bar,
                do_sample,
            ],
            [
                state, chatbot, textbox, imagebox, model_inputs,
                memory_allocated
            ],
        )
        submit_btn.click(
            bot,
            [
                textbox,
                imagebox,
                state,
                prompt,
                ai_prefix,
                user_prefix,
                seperator,
                history_buffer,
                max_new_token_bar,
                num_beams_bar,
                temperature_bar,
                topk_bar,
                topp_bar,
                do_sample,
            ],
            [
                state, chatbot, textbox, imagebox, model_inputs,
                memory_allocated
            ],
        )
        clear_btn.click(clear, [state],
                        [state, chatbot, textbox, imagebox, model_inputs])
    return demo


if __name__ == "__main__":
    llama_path = "checkpoints/llama-7b_hf"
    open_flamingo_path = "checkpoints/OpenFlamingo-9B/checkpoint.pt"
    finetune_path = "checkpoints/mmgpt-lora-v0-release.pt"

    inferencer = Inferencer(
        llama_path=llama_path,
        open_flamingo_path=open_flamingo_path,
        finetune_path=finetune_path)
    init_memory = str(round(torch.cuda.memory_allocated() / 1024**3, 2)) + 'GB'
    demo = build_conversation_demo()
    demo.queue(concurrency_count=3)
    IP = "0.0.0.0"
    PORT = 8997
    demo.launch(server_name=IP, server_port=PORT, share=True)
