import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
# from datasets import load_dataset
from transformers import StoppingCriteriaList
from vxverse.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, CONV_VISION_XVERSE, StoppingCriteriaSub

from vxverse.common.eval_utils import init_model, eval_parser
from vxverse.conversation.conversation import CONV_VISION_Vicuna0, CONV_VISION_XVERSE, CONV_VISION_LLama2
from vxverse.common.config import Config
from vxverse.common.registry import registry



conv_dict = {'pretrain_xverse13b-chat': CONV_VISION_XVERSE}

def read_json(file):
    res = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line.strip()))
    return res

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<ImageHere> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='gqa', help="dataset to evaluate")
parser.add_argument("--gpu_id", type=int, default=0, help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)

model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
CONV_VISION = conv_dict[model_config.model_type]
print("model_config.model_type: {}".format(model_config.model_type))


conv_temp = CONV_VISION.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path



eval_file_path = cfg.evaluation_datasets_cfg["chat"]["eval_file_path"]
img_path = None
batch_size = cfg.evaluation_datasets_cfg["chat"]["batch_size"]
max_new_tokens = cfg.evaluation_datasets_cfg["chat"]["max_new_tokens"]
temperature = cfg.evaluation_datasets_cfg["chat"].get("temperature", 0.5)
top_k = cfg.evaluation_datasets_cfg["chat"].get("top_k", 30)
top_p = cfg.evaluation_datasets_cfg["chat"].get("top_p", 0.85)
do_sample = True
repetition_penalty = cfg.evaluation_datasets_cfg["chat"].get("repetition_penalty", 1.1)

stop_words_ids = [[2]]
stop_sign = "<|endoftext|>"
print("stop_sign", stop_sign)
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id), stopping_criteria=stopping_criteria)

pure_text_datas = read_json(eval_file_path)

count=0
total=0
print_prompt_flag = True
img_list = None
for i, sample in enumerate(tqdm(pure_text_datas)):

    conversations = sample["conversations"]
    conv = conv_temp.copy()


    for j in range(len(conversations)//2):

        conv.append_message(conv.roles[0], conversations[j*2]["value"])

        if print_prompt_flag:
            print("########## Prompts ###########")
            print(conv)
            print_prompt_flag = False

        answer, tokens = chat.answer(conv=conv,
                                  stop_sign=stop_sign,
                                  img_list=img_list,
                                  top_p=top_p,
                                  top_k=top_k,
                                  temperature=temperature,
                                  max_new_tokens=max_new_tokens,
                                  max_length=8192,
                                  repetition_penalty=repetition_penalty,
                                  do_sample=do_sample,)
        print(f"Answer:\n{answer}")
        sample["conversations"][j+1]["value"] = str(answer)



file_save_path = os.path.join(save_path, "open_test_data_lmm_predicts.json")
with open(file_save_path,'w', encoding='utf-8') as f:
    for res in pure_text_datas:
        f.write(json.dumps(res, ensure_ascii=False))
        f.write("\n")





