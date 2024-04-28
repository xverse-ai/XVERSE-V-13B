<div align="center">
<h1>
  XVERSE-V-13B
</h1>
</div>

<p align="center">
        <a href="https://huggingface.co/xverse/XVERSE-V-13B">🤗 Hugging Face</a>&nbsp｜
        <a href="https://modelscope.cn/models/xverse/XVERSE-V-13B/summary" rel="nofollow"><img src="resources/modelscope.png" width="20px" style="max-width: 100%;"> ModelScope</a>&nbsp｜
        <a href="resources/wechat.png">💬 微信社区</a>
</p>

<h4 align="left">
    <p>
        <b>中文</b> |
        <a href="README_EN.md">English</a>
    <p>
</h4>

## 更新信息
- **[2024/04/28]** 发布 **XVERSE-V-13B** 多模态模型。

## 模型介绍

**XVERSE-V-13B** 是由深圳元象科技自主研发的支持图文问答的多模态大模型(Large Multimodal Model)，其主要特点如下：

- **模型结构**：视觉编码器采用了 **openai/clip-vit-large-patch14-224**，文本模型采用了自研的 **XVERSE-13B-Chat** 模型，图像—文本桥接层采用了高效且简洁的两层 **MLP** 结构。
- **训练数据**：图文数据采用的是完全公开的数据集，其中预训练阶段数据量为 2.1B 图文对，微调阶段采用了 8.2M 的指令数据。训练数据几乎全为英文数据，因此模型的能力主要体现在英文方面。
- **图像分辨率**：不同于其他固定图像分辨率的模型，**XVERSE-V-13B** 将图像切分成多个 **224×224** 的块，分别将他们送到视觉模块进行编码，因此能够处理更高分辨率或者不同宽高比的图像，这为我们模型保留了尽可能多的细节信息。
- **训练方式**： **XVERSE-V-13B** 采用了两阶段训练，分别为规模比较大的图文对预训练和规模比较小的指令数据微调。其中预训练阶段，我们冻结❄️视觉模块和 LLM 模块，只训练🔥桥接层部分；
指令微调阶段，我们依然冻结❄️视觉模块和LLM模块，但是微调🔥桥接层部分以及LLM的所有线性层的 LoRA 参数；另外，在指令微调阶段，我们对桥接层部分和 LoRA 部分采用了差分学习率。

## 图像编码示例
对于 448*448 的图像，我们通过 Sliding Window 将其切分成4个局部图像块以及 Resize 得到一个包含全局信息的图像，如下图所示
![avatar](resources/2_2_Trans.drawio.svg)

对于更高分辨率的 448*672 的图像，我们通过 Sliding Window 将其切分成6个局部图像块以及 Resize 得到一个包含全局信息的图像，如下图所示
![avatar](resources/2_3_Trans.drawio.svg)

> <sup>1：Concate* 表示列向量按行进行拼接 </sup> 
> 
> <sup>2：对于其他不同分辨率以及不同宽高比的图像，也是同理进行切块编码 </sup> 

## 评测结果
为了综合评估模型的性能，我们在一系列标准数据集上进行了全面测试，包括 MMBench、MMMU、SEEDBench_IMG、MMStar、LLaVABench、AI2D、ScienceQA、VizWiz、TextVQA、OKVQA 和 GQA 等数据集。这些评估覆盖了模型在多个领域的能力，具体包括 OCR，逻辑推理，关系推理，粗粒度感知和细粒度感知。评估结果如下：

### OpenCompass 榜单
[OpenCompass](https://opencompass.org.cn/home) 是面向大模型评测的一站式平台。 其主要特点如下： 开源可复现：提供公平、公开、可复现的大模型评测方案。因此，我们报告模型在此榜单上的相关结果。

| 数据集                | XVERSE-V-13B | GeminiProVision`*` | Qwen-VL-Plus`*` | Claude-3V Sonnet`*` | LLaVA-Next-Vicuna-13B | Monkey-Chat | OmniLMM-12B | DeepSeek-VL-7B | CogVLM-17B-Chat | TransCore-M | Yi-VL-34B |
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

> <sup>1：带 `*`  号的模型是闭源模型</sup> 

对于上述所有比较模型，我们优先汇报其官方公布的结果。在缺少官方结果的情况下，我们采用了 [OpenCompass 榜单](https://rank.opencompass.org.cn/leaderboard-multimodal)的报告结果。若 OpenCompass 榜单上仍然缺少相应的数据集评估结果，
则来自于我们自行执行的评估流程所获得的数据。而评测框架则采用了[OpenCompass 评估框架](https://github.com/open-compass/OpenCompass/)。

### 传统VQA类任务
传统VQA任务，作为多模态视觉问答领域学术论文常引用的评测任务，具备显著的学术参考价值。因此，我们也将在此类数据集上报告相关的评测结果。

| 数据集                | XVERSE-V-13B | LLaVA-Next-Vicuna-13B | Monkey-Chat | OmniLMM-12B | DeepSeek-VL-7B | CogVLM-17B-Chat | TransCore-M | Yi-VL-34B |
|--------------------|:------------:|:---------------------:| :-------: | :---------: | :--------: |:---------------:|:-----------:| :--------------: |
| ScienceQA          |   **86.4**   |         73.9          |   82.8     |    80.8      |    81.0     |      70.3       |    74.9     |       75.4        |
| OKVQA              |     59.2     |       **60.0**        |   54.7     |    40.8      |    55.1     |      54.4       |    56.7     |       51.4        |
| GQA                |     62.2     |       **65.5**        |   65.4     |    61.1      |    61.8     |      60.5       |    63.6     |       58.3        |
| VizWiz             |   **81.9**   |         54.6          |   75.6     |    64.0      |    50.1     |      44.0       |    41.4     |       70.8        |
| TextVQA            |   **74.2**   |         64.3          |   53.7     |    62.4      |    63.8     |      69.6       |    63.1     |       54.0        |

同理，对于上述所有比较模型，我们优先汇报其官方公布的结果。在缺少官方结果的情况下，则来自于我们自行执行的评估流程所获得的数据。而评测框架则采用了[OpenCompass 评估框架](https://github.com/open-compass/OpenCompass/)。


## 效果示例
这里我们展示全景和细节识别、图表分析、百科解答、教育问答、内容创作和代码生成等能力的样例。

![avatar](resources/Demo_Trans.svg)

## 使用方法

### 环境安装

1. 下载本仓库：

```shell
git clone git@github.com:xverse-ai/XVERSE-V-13B.git
cd XVERSE-V-13B
```

2. 使用 pip 安装依赖：

```shell
pip install -r requirements.txt
```

### 模型准备与加载
1. 模型准备：
我们的模型分为三个部分：视觉编码器 clip-vit-large-patch14-224，大语言模型 XVERSE-13B-Chat 和桥接层 Adapters，这三部分分别可以从下面提供的链接中下载

| XVERSE-13B-Chat                                               | clip-vit-large-patch14-224 |                        Adapters                       |
|---------------------------------------------------------------| :--------------: |:-----------------------------------------------------:|
| <center>[下载](https://huggingface.co/xverse/XVERSE-13B-Chat)   | <center>[下载](https://huggingface.co/openai/clip-vit-large-patch14) |  <center>[下载](https://huggingface.co/xverse/XVERSE-V-13B)|

2. 模型加载：
完成步骤1之后，只需要将模型权重路径填入到配置文件相应的位置中即可：
   1. 对于 clip-vit-large-patch14-224 和 Adapters，请将路径填分别写到 ./eval_configs/vxverse_*.yaml 文件中的 vit_path 和 ckpt 字段中；
   2. 对于XVERSE-13B-Chat，请将路径填写到 ./vxverse/configs/models/vxverse_13bchat.yaml 文件对应的字段中。


### **OKVQA** 和 **GQA** 数据集的测评
1. 数据集准备：
   1. 对于OKVQA测试集可以从<a href="https://okvqa.allenai.org/download.html">从此</a>下载
   2. 对于GQA测试集可以从<a href="https://cs.stanford.edu/people/dorarad/gqa/download.html">从此</a>下载

2. 运行脚本
```shell
python ./eval_vqa.py --cfg-path ./eval_configs/vxverse_hd_benchmark_evaluation.yaml --dataset gqa
```

### 网页 Demo

可通过以下代码启动一个web server，在浏览器输入访问地址后，可对 XVERSE-V-13B 模型进行体验：

```shell
python demo.py --cfg-path ./eval_configs/vxverse_xverse_hd_eval.yaml --gpu-id 0
```

## 特别说明
我们的模型是基于修改并适配后的 [Megatron](https://github.com/NVIDIA/Megatron-LM) 框架训练的，而 Pytorch 框架下的模型加载，demo 体验和数据集的评估则是基于[MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)代码修改而来的。


## 局限性与免责申明

XVERSE-V-13B 与其它所有 LMM 一样，在某些情况下可能会产生不准确、有偏见或其他令人反感的内容。因此，请谨慎使用模型生成的内容，请勿将生成的有害内容进行传播，在部署任何 XVERSE-V-13B 的应用之前，开发人员应根据其具体应用对模型进行安全测试和调优。

我们强烈警告不要将 XVERSE-V-13B 模型用于制造或传播有害信息，或进行任何可能损害公众、国家、社会安全或违反法规的活动。如果使用 XVERSE-V-13B 模型产生任何问题，无论是数据安全问题、公共舆论风险，还是模型被误解、滥用、传播或不合规使用所引发的任何风险和问题，我们将不承担任何责任。

## 模型开源协议

使用本仓库的源码需要遵循 [Apache-2.0](LICENSE) 开源协议，使用 XVERSE-V-13B 的模型权重则需要遵循[模型许可协议](MODEL_LICENSE.pdf)。

XVERSE-V-13B 模型权重对学术研究**完全开放**，并且支持**免费商用**。如需申请商业许可证，请填写【[申请表](https://chat.xverse.cn/home/business.html)】，如有其他问题或合作，请联系 <opensource@xverse.cn>。

