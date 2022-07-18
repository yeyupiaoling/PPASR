# PPASR流式与非流式语音识别

![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/PPASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PPASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/PPASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

本项目将分三个阶段分支，分别是[入门级](https://github.com/yeyupiaoling/PPASR/tree/%E5%85%A5%E9%97%A8%E7%BA%A7) 、[进阶级](https://github.com/yeyupiaoling/PPASR/tree/%E8%BF%9B%E9%98%B6%E7%BA%A7) 和[最终级](https://github.com/yeyupiaoling/PPASR) 分支，当前为最终级，持续维护版本。PPASR中文名称PaddlePaddle中文语音识别（PaddlePaddle Automatic Speech Recognition），是一款基于PaddlePaddle实现的语音识别框架，PPASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。

**别忘了star**

## 在线使用

**1. [在AI Studio平台训练预测](https://aistudio.baidu.com/aistudio/projectdetail/3290199)**

<!-- **2. [在线使用Dome](https://ppasr.yeyupiaoling.cn)** -->

<br/>

**本项目使用的环境：**
 - Anaconda 3
 - Python 3.7
 - PaddlePaddle 2.2.0
 - Windows 10 or Ubuntu 18.04

## 更新记录

 - 2022.07.12: 完成GUI界面的录音实时识别。
 - 2022.06.14: 支持`deepspeech2_big`模型，适合WenetSpeech大数据集训练模型。
 - 2022.01.16: 支持多种预处理方法。
 - 2022.01.15: 支持英文语音识别。
 - 2022.01.13: 支持给识别结果加标点符号。
 - 2021.12.23: 支持pip安装。
 - 2021.11.30: 全面修改为流式语音识别模型。
 - 2021.11.09: 增加制作WenetSpeech数据集脚本和文档。
 - 2021.10.10: 提供三个公开数据集的DeepSpeech2预训练模型下载。
 - 2021.09.30: 在导出模型时，把归一化放在模型用，推理时直接在模型中完成数据归一化，不需要额外对数据归一化再输入到网络模型中。
 - 2021.09.18: 初步完成基本程序。

## 视频讲解
[知识点讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Rr4y1D7iZ)

## 模型下载
|                                            数据集                                            |    使用模型     | 语言  |                                     解码参数                                      | 测试集字错率（词错率） |                              下载地址                               |
|:-----------------------------------------------------------------------------------------:|:-----------:|:---:|:-----------------------------------------------------------------------------:|:-----------:|:---------------------------------------------------------------:|
|             [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)             | deepspeech2 | 中文  | alpha=2.2<br>beta=4.3<br>beam_size=300<br>cutoff_prob=0.99<br>cutoff_top_n=40 |  0.077042   | [点击下载](https://download.csdn.net/download/qq_33200967/29121153) |
| [free st chinese mandarin corpus](https://openslr.magicdatatech.com/resources/38) (109小时) | deepspeech2 | 中文  | alpha=2.2<br>beta=4.3<br>beam_size=300<br>cutoff_prob=0.99<br>cutoff_top_n=40 |  0.137442   | [点击下载](https://download.csdn.net/download/qq_33200967/30296023) |
|             [thchs_30](https://openslr.magicdatatech.com/resources/18) (34小时)             | deepspeech2 | 中文  | alpha=2.2<br>beta=4.3<br>beam_size=300<br>cutoff_prob=0.99<br>cutoff_top_n=40 |  0.062654   | [点击下载](https://download.csdn.net/download/qq_33200967/26929682) |
|                             超大数据集(1600多小时真实数据)+(1300多小时合成数据)                              | deepspeech2 | 中文  | alpha=2.2<br>beta=4.3<br>beam_size=300<br>cutoff_prob=0.99<br>cutoff_top_n=40 |  0.056835   | [点击下载](https://download.csdn.net/download/qq_33200967/58036573) |
|           [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时)           | deepspeech2 | 英文  | alpha=1.9<br>beta=0.3<br>beam_size=500<br>cutoff_prob=1.0<br>cutoff_top_n=40  |   0.10855   | [点击下载](https://download.csdn.net/download/qq_33200967/77978970) |

**说明：** 
1. 这里字错率是使用`eval.py`程序并使用集束搜索解码`ctc_beam_search`方法计算得到的。
2. 除了aishell数据集按照数据集本身划分的训练数据和测试数据，其他的都是按照项目设置的固定比例划分训练数据和测试数据。
3. 下载的压缩文件已经包含了`mean_std.npz`和`vocabulary.txt`，需要把解压得到的全部文件复制到项目根目录下。

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/PPASR/issues) 交流


## 文档教程

- [快速安装](./docs/install.md)
- [快速使用](./docs/GETTING_STARTED.md)
- [数据准备](./docs/dataset.md)
- [WenetSpeech数据集](./docs/wenetspeech.md)
- [合成语音数据](./docs/generate_audio.md)
- [数据增强](./docs/augment.md)
- [训练模型](./docs/train.md)
- [集束搜索解码](./docs/beam_search.md)
- [执行评估](./docs/eval.md)
- [导出模型](./docs/export_model.md)
- [使用标点符号模型](./docs/punctuation.md)
- 预测
   - [本地模型](./docs/infer.md)
   - [长语音模型](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)
   - [Nvidia Jetson部署](./docs/nvidia-jetson.md)


## 快速预测

 - 下载作者提供的模型或者训练模型，然后执行[导出模型](./docs/export_model.md)，使用`infer_path.py`预测音频，通过参数`--wav_path`指定需要预测的音频路径，完成语音识别，详情请查看[模型部署](./docs/infer.md)。
```shell script
python infer_path.py --wav_path=./dataset/test.wav
```

输出结果：
```
-----------  Configuration Arguments -----------
alpha: 1.2
beam_size: 10
beta: 0.35
cutoff_prob: 1.0
cutoff_top_n: 40
decoding_method: ctc_greedy
enable_mkldnn: False
is_long_audio: False
lang_model_path: ./lm/zh_giga.no_cna_cmn.prune01244.klm
mean_std_path: ./dataset/mean_std.npz
model_dir: ./models/infer/
to_an: True
use_gpu: True
use_tensorrt: False
vocab_path: ./dataset/zh_vocab.txt
wav_path: ./dataset/test.wav
------------------------------------------------
消耗时间：132, 识别结果: 近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 94
```


 - 长语音预测

```shell script
python infer_path.py --wav_path=./dataset/test_vad.wav --is_long_audio=True
```


 - Web部署

![录音测试页面](./docs/images/infer_server.jpg)


 - GUI界面部署

![GUI界面](./docs/images/infer_gui.jpg)


## 相关项目
 - 基于PaddlePaddle实现的声纹识别：[VoiceprintRecognition-PaddlePaddle](https://github.com/yeyupiaoling/VoiceprintRecognition-PaddlePaddle)
 - 基于PaddlePaddle静态图实现的语音识别：[PaddlePaddle-DeepSpeech](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech)
 - 基于Pytorch实现的语音识别：[MASR](https://github.com/yeyupiaoling/MASR)


## 特别感谢

 - 感谢 <img src="docs/images/PyCharm_icon.png" height="25" width="25" >[JetBrains开源社区](https://jb.gg/OpenSourceSupport) 提供开发工具。

## 参考资料
 - https://github.com/PaddlePaddle/PaddleSpeech
 - https://github.com/jiwidi/DeepSpeech-pytorch
 - https://github.com/wenet-e2e/WenetSpeech
