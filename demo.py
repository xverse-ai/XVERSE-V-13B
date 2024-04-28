import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from transformers import StoppingCriteriaList

from vxverse.common.config import Config
from vxverse.common.registry import registry
from vxverse.conversation.conversation import Chat, CONV_VISION_XVERSE, StoppingCriteriaSub

# imports modules for registration

# from vxverse.datasets.builders import *
# from vxverse.models import *
# from vxverse.processors import *
# from vxverse.runners import *
# from vxverse.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--server_port", type=int, default=20029, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_xverse13b-chat': CONV_VISION_XVERSE}

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]
# CONV_VISION.system = ""
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

if "vicuna" in model_config.model_type:
    stop_sign = "###"
    stop_words_ids = [[835], [2277, 29937]]
elif "xverse" in model_config.model_type:
    stop_words_ids = [[2]]
    stop_sign = "<|endoftext|>"
elif "llama" in model_config.model_type:
    stop_sign = "</s>"
    stop_words_ids = [[2]]
else:
    raise ValueError("Not support model type.")

print("stop_sign", stop_sign)
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria, vis_processor_name=vis_processor_cfg.name)
print('Initialization Finished')


# ========================================
#             Gradio Setting
# ========================================

def escape_markdown(text):
    # List of Markdown special characters that need to be escaped
    md_chars = ['<', '>']
    # Escape each special character
    for char in md_chars:
        text = text.replace(char, '\\' + char)
    text = text.replace("\u200b\n", "")
    return text

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Upload your image and chat', interactive=True), chat_state, img_list


def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    chat.encode_img(img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list




def gradio_ask(user_message, chatbot, chat_state, gr_img, img_list, upload_flag, replace_flag):
    if len(user_message) == 0:
        text_box_show = 'Input should not be empty!'
    else:
        text_box_show = ''
    if isinstance(gr_img, dict):
        gr_img, mask = gr_img['image'], gr_img['mask']
    else:
        mask = None

    if chat_state is None:
        chat_state = CONV_VISION.copy()

    if upload_flag:
        if replace_flag:
            chat_state = CONV_VISION.copy()  # new image, reset everything
            replace_flag = 0
            chatbot = []
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        upload_flag = 0
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]

    return text_box_show, chatbot, chat_state, img_list, upload_flag, replace_flag


def gradio_answer(chatbot, chat_state, img_list, top_p, temperature):
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    llm_message = chat.answer(conv=chat_state,
                              stop_sign=stop_sign,
                              img_list=img_list,
                              top_p=top_p,
                              temperature=temperature,
                              max_new_tokens=500,
                              max_length=2048)[0]
    chatbot[-1][1] = llm_message
    print("##############")
    print("chat state:{}".format(chat_state))
    return chatbot, chat_state, img_list


def gradio_stream_answer(chatbot, chat_state, img_list, top_p, temperature, do_sample=False):
    # if img_list != None  To support pure text conversation
    if img_list != None and len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            chat.encode_img(img_list)
    streamer = chat.stream_answer(conv=chat_state,
                                  img_list=img_list,
                                  temperature=temperature,
                                  top_p=top_p,
                                  do_sample=do_sample,
                                  max_new_tokens=2048,
                                  max_length=8192)

    output = ''
    for new_output in streamer:
        escapped = escape_markdown(new_output)
        output += escapped
        output = output.split(stop_sign)[0]  # remove the stop sign
        output = output.split('Assistant:')[-1]
        chatbot[-1][1] = output
        yield chatbot, chat_state, img_list
    chat_state.messages[-1][1] = output
    print("##############")
    print("chat_state:{}".format(chat_state))
    return chatbot, chat_state, img_list


def image_upload_trigger(upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list:
        replace_flag = 1
    return upload_flag, replace_flag

def example_trigger(image, text_input, upload_flag, replace_flag, img_list):
    # set the upload flag to true when receive a new image.
    # if there is an old image (and old conversation), set the replace flag to true to reset the conv later.
    upload_flag = 1
    if img_list or replace_flag == 1:
        replace_flag = 1

    return upload_flag, replace_flag

title = """<h1 align="center">Demo of XVERSE-V</h1>"""
description = """<h3>This is the demo of XVERSE-V. Upload your images and start chatting!</h3>"""


#TODO show examples below

text_input = gr.Textbox(placeholder='Upload your image and chat', interactive=True, show_label=False, container=False,
                        scale=8)
with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil")
            clear = gr.Button("Restart")
            
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1,
                value=0.8,
                step=0.05,
                interactive=True,
                label="Top P",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            do_sample = gr.inputs.Checkbox(label="do_sample")  # default False
        with gr.Column(scale=2):
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='Visual-XVERSE')

            with gr.Row():
                text_input.render()
                send = gr.Button("Send", variant='primary', size='sm', scale=1)

    upload_flag = gr.State(value=0)
    replace_flag = gr.State(value=0)
    image.upload(image_upload_trigger, [upload_flag, replace_flag, img_list], [upload_flag, replace_flag])


    text_input.submit(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
    ).success(
        gradio_stream_answer,
        [chatbot, chat_state, img_list, top_p, temperature, do_sample],
        [chatbot, chat_state]
    )

    send.click(
        gradio_ask,
        [text_input, chatbot, chat_state, image, img_list, upload_flag, replace_flag],
        [text_input, chatbot, chat_state, img_list, upload_flag, replace_flag], queue=False
    ).success(
        gradio_stream_answer,
        [chatbot, chat_state, img_list, top_p, temperature, do_sample],
        [chatbot, chat_state]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, chat_state, img_list], queue=False)

demo.queue(concurrency_count=4)
demo.launch(share=False, server_name="0.0.0.0", server_port=args.server_port, enable_queue=True)
# demo.launch(share=False, server_name="0.0.0.0", server_port=args.server_port, max_threads=4)
