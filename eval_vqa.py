import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from vxverse.datasets.datasets.vqa_datasets import OKVQAEvalData, GQAEvalData
from vxverse.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from vxverse.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from vxverse.common.eval_utils import prepare_texts, init_model, eval_parser
from vxverse.conversation.conversation import CONV_VISION_XVERSE
from vxverse.common.config import Config
from vxverse.common.registry import registry


conv_dict = {'pretrain_xverse13b-chat': CONV_VISION_XVERSE}


stop_words_ids = [[2]]
do_sample = False


def collater4hd(samples):
    # image, question, question_id, img_id
    batch_images, batch_questions, batch_question_ids, batch_img_ids = [], [], [], []
    batch_patches_per_image, batch_total_images, batch_labels = [], [], []
    for sample in samples:
        if not isinstance(sample["image"], list):
            sample["image"] = [sample["image"]]
        patches_per_image = []
        for img in sample["image"]:
            patches_per_image.append(img.shape[0])
        batch_patches_per_image.append(patches_per_image)
        batch_total_images.append(len(sample["image"]))

    for sample in samples:
        batch_images.append(torch.cat(sample["image"], dim=0))
        batch_questions.append(sample["question"])
        batch_question_ids.append(sample.get("question_id", 0))
        batch_img_ids.append(sample.get("img_id", 0))
        batch_labels.append(sample.get("label", None))
    return {
        "image": batch_images,
        "question": batch_questions,
        "question_id": batch_question_ids,
        "img_id": batch_img_ids,
        "label":batch_labels,
        "patches_per_image": batch_patches_per_image,
        "total_images": batch_total_images
    }


def list_of_str(arg):
    return list(map(str, arg.split(',')))


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='gqa', help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)

model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)
CONV_VISION = conv_dict[model_config.model_type]
proce_type = list(cfg.datasets_cfg.keys())[0]
vis_proce_type = cfg.datasets_cfg.get(proce_type).vis_processor.train.name
print("model_config.model_type: {}".format(model_config.model_type))
print("vision process type: {}".format(vis_proce_type))

conv_temp = CONV_VISION.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path



if 'okvqa' in args.dataset:

    print("##################  OKVQA EVAL ###############")
    eval_file_path = cfg.evaluation_datasets_cfg["okvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["okvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["okvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["okvqa"]["max_new_tokens"]

    print_res_flag = True
    evaluation_annntation_path = os.path.join(eval_file_path, "okvqa_test_split.json")
    with open(evaluation_annntation_path) as f:
        ok_vqa_test_split = json.load(f)


    data = OKVQAEvalData(ok_vqa_test_split, vis_processor, img_path)
    if vis_proce_type == "hd_image_train":
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collater4hd)
    else:
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    vxverse_predict = []

    for i, samples in enumerate(tqdm(eval_dataloader)):
        images, questions, question_ids, img_ids = samples["image"], samples["question"], samples["question_id"], samples["img_id"]
        patches_per_images = samples.get("patches_per_image", None)
        total_images = samples.get("total_images", None)
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        texts = [text.lstrip() for text in texts]

        answers = model.generate(images, texts, patches_per_images=patches_per_images, max_new_tokens=max_new_tokens, do_sample=False, stop_words_ids=stop_words_ids)

        for answer, question_id, question, img_id, text in zip(answers, question_ids, questions, img_ids, texts):
            result = dict()

            answer = answer.lower()

            result['answer'] = answer
            result['question_id'] = int(question_id)
            result["Prompt"] = text
            vxverse_predict.append(result)
        if i % 10 == 0:
            print(vxverse_predict[i])


    file_save_path= os.path.join(save_path,"okvqa.json")
    with open(file_save_path,'w', encoding='utf-8') as f:
        for res in vxverse_predict:
            f.write(json.dumps(res, ensure_ascii=False))
            f.write("\n")

    annFile = os.path.join(eval_file_path,"mscoco_val2014_annotations_clean.json")
    quesFile = os.path.join(eval_file_path,"OpenEnded_mscoco_val2014_questions_clean.json" )

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall OKVQA Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)



if 'gqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["gqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["gqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["gqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["gqa"]["max_new_tokens"]
    gqa = json.load(open(eval_file_path))
    data = GQAEvalData(gqa, vis_processor, img_path)
    if vis_proce_type == "hd_image_train":
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collater4hd)
    else:
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0
    print_prompt_flag = True
    vxverse_predict = []
    for i, samples in enumerate(tqdm(eval_dataloader)):
        images, texts, labels = samples["image"], samples["question"], samples["label"]
        patches_per_images = samples.get("patches_per_image", None)
        total_images = samples.get("total_images", None)
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        texts = [text.lstrip() for text in texts]

        if print_prompt_flag:
            print("########## Prompts ###########")
            print(texts)
            print_prompt_flag = False
        answers = model.generate(images, texts, patches_per_images=patches_per_images, max_new_tokens=max_new_tokens, do_sample=do_sample, stop_words_ids=stop_words_ids)

        for answer, label, text in zip(answers, labels, texts):
            result = dict()
            answer = answer.lower()
            result['pred'] = answer
            result['gt'] = label
            result['Prompt'] = text
            vxverse_predict.append(result)
            if label in answer.lower():
                count+=1
            total+=1
        if i % 20 == 0:
            print(vxverse_predict[i])

        print("acc count:", count)

    print('gqa val:', count / total * 100, flush=True)

    file_save_path = os.path.join(save_path, "gqa.json")
    with open(file_save_path,'w', encoding='utf-8') as f:

        for res in vxverse_predict:
            f.write(json.dumps(res, ensure_ascii=False))
            f.write("\n")


