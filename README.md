# PPASR语音识别（最终级）

![License](https://img.shields.io/badge/license-Apache%202-red.svg)
![python version](https://img.shields.io/badge/python-3.7+-orange.svg)
![support os](https://img.shields.io/badge/os-linux-yellow.svg)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PPASR?style=social)

本项目将分三个阶段分支，分别是[入门级](https://github.com/yeyupiaoling/PPASR/tree/%E5%85%A5%E9%97%A8%E7%BA%A7) 、[进阶级](https://github.com/yeyupiaoling/PPASR/tree/%E8%BF%9B%E9%98%B6%E7%BA%A7) 和[最终级](https://github.com/yeyupiaoling/PPASR) 分支，当前为最终级，持续维护版本。

PPASR（最终级）基于PaddlePaddle2实现的端到端自动语音识别，相比进阶级，最终级完善了部署上，使该项目能够在各个设备上部署使用。

本项目使用的环境：
 - Anaconda 3
 - Python 3.7
 - PaddlePaddle 2.1.3
 - Windows 10 or Ubuntu 18.04

## 更新记录

 - 2021.09.18: 初步完成基本程序。

## 模型下载
| 数据集 | 卷积层数量 | 循环神经网络的数量 | 循环神经网络的大小 | 测试集字错率 | 下载地址 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| aishell(179小时) | 2 | 3 | 512 | 训练中 | [训练中]() |
| free_st_chinese_mandarin_corpus(109小时) | 2 | 3 | 512 | 训练中 | [训练中]() |
| thchs_30(34小时) | 2 | 3 | 512 | 训练中 | [训练中]() |
| 超大数据集(1600多小时真实数据)+(1300多小时合成数据) | 2 | 3 | 512 | 训练中 | [训练中]() |

**说明：** 这里提供的是训练参数，如果要用于预测，还需要执行[导出模型](./docs/export_model.md)，使用的解码方法是集束搜索。

>有问题欢迎提 [issue](https://github.com/yeyupiaoling/PPASR/issues) 交流


## 文档教程

- [快速安装](./docs/install.md)
- [数据准备](./docs/dataset.md)
- [合成语音数据](./docs/generate_audio.md)
- [数据增强](./docs/augment.md)
- [训练模型](./docs/train.md)
- [集束搜索解码](./docs/beam_search.md)
- [执行评估](./docs/eval.md)
- [导出模型](./docs/export_model.md)
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
 - 基于PaddlePaddle静态图实现的语音识别：[PPASR](https://github.com/yeyupiaoling/PaddlePaddle-DeepSpeech)
 - 基于Pytorch实现的语音识别：[MASR](https://github.com/yeyupiaoling/MASR)
