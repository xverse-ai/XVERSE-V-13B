<div align="center">
<h1>
  XVERSE-V-13B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-V-13B">🤗 Hugging Face</a>&nbsp｜
        <a href="https://modelscope.cn/models/xverse/XVERSE-V-13B/summary" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜
        <a href="resources/wechat.png">💬 WeChat</a>
</p>

<h4 align="left">
    <p>
        <a href="README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

## Update Information
- **[2024/04/28]** release **XVERSE-V-13B** large multimodal model.

## Model Introduction

**XVERSE-V-13B**, a large multimodal model, is independently developed by Shenzhen Yuanxiang Technology. Its key features are as follows:

- **Model Structure**: The visual encoder adopts **openai/clip vit target patch14-224**, the text model adopts the self-developed **XVERSE-13B Chat** model, and the image bridging layer adopts an efficient and concise two-layer **MLP** structure.
- **Training Data**: The multimodal data utilized in the training process is sourced from fully open datasets. During the pre-training stage, the dataset comprises 2.1 billion pairs of images and text, while the fine-tuning stage employs 8.2 million instruction data points. The training data is predominantly composed of English data, hence the model's proficiency primarily lies in the English domain.
- **Image Resolution**: Unlike other models with fixed image resolutions, XVERSE-V-13B divides images into multiple 224×224 blocks, each sent to the visual module for encoding. This allows it to handle higher-resolution images without the need for cropping.
- **Training Schedule**: **XVERSE-V-13B** adopts a two-stage training approach, consisting of a large-scale multimodal pre-training followed by fine-tuning on a smaller-scale instruction dataset. During the pre-training stage, we freeze ❄️ the visual and LLM  modules and only train 🔥 the bridging layer; 
During the instruction fine-tuning stage, we still freeze ❄️ the visual and LLM modules, but fine-tune 🔥 the bridging layer as well as the LoRA parameters of all linear layers in the LLM. Additionally, during the instruction fine-tuning stage, 
we apply differential learning rates to the bridging layer and the LoRA components.


## Example of image encoding
For a 448×448 image, we split it into 4 local image blocks using Sliding Window and resize it to obtain a global information-containing image, as shown in the figure below.
![avatar](resources/2_2_Trans.drawio.svg)

For a higher resolution 448×672 image, we split it into 6 local image blocks using Sliding Window and resize it to obtain a global information-containing image, as shown in the figure below.
![avatar](resources/2_3_Trans.drawio.svg)

> <sup>1: Concate* represents concatenation of column vectors row-wise. </sup>
> 
> <sup>2: For other images with different resolutions and aspect ratios, the same chunk encoding process applies. </sup> 



## Evaluation Reports

To comprehensively assess the model's performance, we conducted thorough testing on a series of standard datasets, including MMBench, MMMU, SEEDBench_IMG, MMStar, LLaVABench, AI2D, ScienceQA, VizWiz, TextVQA, OKVQA, and GQA, among others. These evaluations span the model's capabilities across various domains, encompassing OCR, logical reasoning, relational reasoning, coarse-grained perception, and fine-grained perception. The evaluation results are as follows:

### OpenCompass Leaderboard
[OpenCompass](https://opencompass.org.cn/home) is a one-stop platform for large-scale model evaluation. Its main features are as follows: Open Source and Reproducible: It provides a fair, open, and reproducible evaluation framework for large-scale models. Therefore, we report the relevant results of our model on this leaderboard.

| Datasets           | XVERSE-V-13B | GeminiProVision`*` | Qwen-VL-Plus`*` | Claude-3V Sonnet`*` | LLaVA-Next-Vicuna-13B | Monkey-Chat | OmniLMM-12B | DeepSeek-VL-7B | CogVLM-17B-Chat | TransCore-M | Yi-VL-34B |
|--------------------|:------------:|:------------------:|:---------------:|:-------------------:|:---------------------:|:-----------:|:-----------:|:--------------:|:---------------:|:-----------:|:---------:|
| MMBench            |     75.6     |        73.6        |      67.0       |        67.8         |         70.0          |    72.4     |    71.7     |      73.8      |      65.8       |  **82.3**   |   72.4    |
| MMBench-CN         |     74.7     |        74.3        |      70.7       |        64.2         |         68.5          |    67.5     |    62.0     |      71.4      |      55.9       |  **80.7**   |   70.7    |
| MMStar             |   **47.8**   |        38.6        |      39.7       |        44.2         |         40.4          |    40.7     |    39.6     |      40.5      |      39.9       |    35.6     |   40.5    |
| MMMU-Val           |     43.3     |      **48.9**      |      39.8       |        47.4         |         37.3          |    40.7     |    41.8     |      38.3      |      37.3       |    41.0     |   45.1    |
| MathVistaMini-Test |     44.1     |      **46.5**      |      37.6       |        45.0         |         34.1          |    35.9     |    34.7     |      36.9      |      35.0       |    32.3     |   31.5    |
| HallusionBench     |     31.8     |      **45.2**      |      40.6       |        41.3         |         31.8          |    39.3     |    35.8     |      34.5      |      35.4       |    27.3     |   35.3    |
| AI2D-Test          |     70.4     |        70.2        |      65.7       |        69.9         |       **72.2**        |    68.5     |    63.3     |      65.3      |      63.3       |    64.1     |   65.9    |
| OCRBench           |     489      |       680.0        |    **726.0**    |        646.0        |         537.0         |    534.0    |    420.0    |     435.0      |      590.0      |    405.0    |   290.0   |
| SEEDBench_IMG      |   **72.4**   |        70.7        |      65.7       |        65.0         |         71.4          |    68.9     |    71.5     |      70.1      |      68.8       |    72.0     |   68.1    |
| LLaVABench         |   **82.3**   |        79.9        |      73.7       |        73.2         |         73.9          |    60.5     |    75.8     |      77.8      |      73.9       |    66.8     |   62.3    |

> <sup>1. Models marked with an asterisk `*` are closed-source models.</sup>

For all the compared models mentioned above, we prioritize reporting their officially published results. In cases where official results are unavailable, we rely on the reported results from the [OpenCompass leaderboard](https://rank.opencompass.org.cn/leaderboard-multimodal). 
If the corresponding dataset evaluation results are still missing from the OpenCompass leaderboard, we include data obtained from our own evaluation process. 
The evaluation framework used adheres to the [OpenCompass evaluation framework](https://github.com/open-compass/OpenCompass/).

### Traditional VQA tasks
The traditional Visual Question Answering (VQA) task, frequently referenced in academic literature in the field of multimodal visual question answering, holds significant academic reference value. 
Therefore, we will also report relevant evaluation results on datasets of this kind.


| Datasets  | XVERSE-V-13B | LLaVA-Next-Vicuna-13B | Monkey-Chat | OmniLMM-12B | DeepSeek-VL-7B | CogVLM-17B-Chat | TransCore-M | Yi-VL-34B |
|-----------|:------------:|:---------------------:| :-------: | :---------: | :--------: |:---------------:|:-----------:| :--------------: |
| ScienceQA |   **86.4**   |         73.9          |   82.8     |    80.8      |    81.0     |      70.3       |    74.9     |       75.4        |
| OKVQA     |     59.2     |       **60.0**        |   54.7     |    40.8      |    55.1     |      54.4       |    56.7     |       51.4        |
| GQA       |     62.2     |       **65.5**        |   65.4     |    61.1      |    61.8     |      60.5       |    63.6     |       58.3        |
| VizWiz    |   **81.9**   |         54.6          |   75.6     |    64.0      |    50.1     |      44.0       |    41.4     |       70.8        |
| TextVQA   |   **74.2**   |         64.3          |   53.7     |    62.4      |    63.8     |      69.6       |    63.1     |       54.0        |


Similarly, for all the compared models mentioned above, we prioritize reporting their officially published results. In the absence of official results, data is obtained from our own evaluation process. 
The evaluation framework used adheres to the [OpenCompass evaluation framework](https://github.com/open-compass/OpenCompass/).


## Demo 
Here we present examples of abilities such as panoramic and detail recognition, content creation, travel assistant, chart analysis, educational Q&A, and code generation.

![avatar](resources/Demo_Trans.svg)


## Usage

### Environment Setup

1. Clone this repository:

```shell
git clone git@github.com:xverse-ai/XVERSE-V-13B.git
cd XVERSE-V-13B
```

2. Install the dependencies using pip:

```shell
pip install -r requirements.txt
```

### Model preparation and loading
1. Model preparation:
Our model includes three parts: the visual encoder (clip-vit-large-patch14-224), the large language model XVERSE-13B-Chat, and the bridge layer. These three parts can be downloaded from the links provided below

| XVERSE-13B-Chat                                             | clip-vit-large-patch14-224 |                            Adapters                            |
|-------------------------------------------------------------| :--------------: |:--------------------------------------------------------------:|
| <center>[Download](https://huggingface.co/xverse/XVERSE-13B-Chat)  | <center>[Download](https://huggingface.co/openai/clip-vit-large-patch14) | <center>[Download](https://huggingface.co/xverse/XVERSE-V-13B) |


2. Once complete step 1, you can simply fill the model weight path into the corresponding location in the configuration file:
   1. for clip-vit-large-patch14-224 and Adapters, please set the path to the right key of ./eval_configs/vxverse_*.yaml 
   2. for XVERSE-13B-Chat, please set the path to the right key of ./vxverse/configs/models/vxverse_13bchat.yaml


### Evaluation of **OKVQA** and **GQA** datasets
1. Prepare the dataset
   1. The OKVQA test set can be accessed from <a href="https://okvqa.allenai.org/download.html">here</a>.
   2. The GQA test set can be accessed from <a href="https://cs.stanford.edu/people/dorarad/gqa/download.html">here</a>.


2. running
```shell
python ./eval_vqa.py --cfg-path ./eval_configs/vxverse_hd_benchmark_evaluation.yaml --dataset gqa
```


### Web Demo


You can start a web server with the following code. After entering the access address in your browser, you can experience the **XVERSE-V-13B** model.

```shell
python demo.py --cfg-path ./eval_configs/vxverse_xverse_hd_eval.yaml --gpu-id 0
```

## Notes
Our training infrastructure is based on [Megatron](https://github.com/NVIDIA/Megatron-LM) and the demo and model loading under the PyTorch framework derives from [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4).

## Limitations and Disclaimer

Like all other Large Language Models (LLMs), XVERSE-V-13B may produce inaccurate, biased, or otherwise offensive content under certain circumstances. Therefore, please use the content generated by the model with caution and refrain from disseminating harmful content. Before deploying any application of XVERSE-V-13B, developers should conduct safety tests and optimization of the model according to its specific application.

We strongly warn against the use of the XVERSE-V-13B model for producing or spreading harmful information, or conducting any activities that might harm the public, national, or social security, or violate regulations. We assume no responsibility for any problems arising from the use of the XVERSE-V-13B model, whether it be data security issues, public opinion risks, or any risks and issues caused by misunderstanding, misuse, dissemination, or non-compliance with the model.

## Open Source License

The use of the source code in this repository must follow the [Apache-2.0](LICENSE) open-source license, while the use of the model weights of XVERSE-V-13B needs to adhere to the [Model License Agreement](MODEL_LICENSE.pdf).

The XVERSE-V-13B model weights are **fully open** to academic research and support **free commercial use**.  To apply for a commercial license, please fill in the [application form](https://chat.xverse.cn/home/business.html). For other questions or collaborations, please contact <opensource@xverse.cn>.

