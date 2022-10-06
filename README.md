![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/PPASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PPASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/PPASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# PPASR流式与非流式语音识别项目

本项目将分三个阶段分支，分别是[入门级](https://github.com/yeyupiaoling/PPASR/tree/%E5%85%A5%E9%97%A8%E7%BA%A7) 、[进阶级](https://github.com/yeyupiaoling/PPASR/tree/%E8%BF%9B%E9%98%B6%E7%BA%A7) 和[最终级](https://github.com/yeyupiaoling/PPASR) 分支，当前为最终级，持续维护版本。PPASR中文名称PaddlePaddle中文语音识别（PaddlePaddle Automatic Speech Recognition），是一款基于PaddlePaddle实现的语音识别框架，PPASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。**别忘了star**

**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`1169600237`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>

## 在线使用

**1. [在AI Studio平台训练预测](https://aistudio.baidu.com/aistudio/projectdetail/3290199)**

<!-- **2. [在线使用Dome](https://ppasr.yeyupiaoling.cn)** -->

<br/>

**本项目使用的环境：**
 - Anaconda 3
 - Python 3.8
 - PaddlePaddle 2.3.2
 - Windows 10 or Ubuntu 18.04


## 项目快速了解

 1. 本项目支持流式识别模型`deepspeech2`、`deepspeech2_big`，非流式模型`deepspeech2_no_stream`、`deepspeech2_big_no_stream`。
 2. 本项目支持两种解码器，分别是集束搜索解码器`ctc_beam_search`和贪心解码器`ctc_greedy`，集束搜索解码器`ctc_beam_search`准确率更高，但不支持Windows。

## 更新记录

 - 2022.10.01: 调整数据预处理，此前下载的模型，需要重新下载。
 - 2022.09.18: 支持使用WebSocket调用流式识别。
 - 2022.08.26: 修改使用kaldi实现`fbank`和`mfcc`预处理方法。
 - 2022.08.22: 增加非流式模型`deepspeech2_no_stream`和`deepspeech2_big_no_stream`。
 - 2022.08.04: 发布1.0版本，优化实时识别流程。
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

 - [知识点讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Rr4y1D7iZ)
 - [流式识别的使用讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Te4y1h7KK)

## 模型下载

本项目支持流式识别模型`deepspeech2`、`deepspeech2_big`，非流式模型`deepspeech2_no_stream`、`deepspeech2_big_no_stream`。

|         使用模型          |                                  数据集                                  | 预处理方式 | 参数大小（M）`*` | 语言  |     测试集字错率（词错率）      |                               下载地址                               |
|:---------------------:|:---------------------------------------------------------------------:|:-----:|:----------:|:---:|:--------------------:|:----------------------------------------------------------------:|
|    deepspeech2_big    |            [WenetSpeech](./docs/wenetspeech.md) (10000小时)             | fbank |    167     | 中文  | 0.07392(AIShell的测试集) | [点击下载](https://pan.baidu.com/s/1DDU92HGH3RRMySBfroXz5w?pwd=ai6p) |
|      deepspeech2      |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank |     35     | 中文  |       0.07280        | [点击下载](https://pan.baidu.com/s/1KFRDIxxlW092Ad70-TNKlw?pwd=m0e0) |
|    deepspeech2_big    |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank |    167     | 中文  |       0.05370        | [点击下载](https://pan.baidu.com/s/1KFRDIxxlW092Ad70-TNKlw?pwd=m0e0) |
| deepspeech2_no_stream |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank |     98     | 中文  |       0.07253        | [点击下载](https://pan.baidu.com/s/1KFRDIxxlW092Ad70-TNKlw?pwd=m0e0) |
|      deepspeech2      | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank |     35     | 英文  |       0.16369        | [点击下载](https://download.csdn.net/download/qq_33200967/77978970)  | 
|    deepspeech2_big    | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank |    167     | 英文  |       0.12779        | [点击下载](https://pan.baidu.com/s/1xfVPDuOAA3rc_6_1JaR5QQ?pwd=673u) | 
| deepspeech2_no_stream | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank |     98     | 英文  |       0.09131        | [点击下载](https://pan.baidu.com/s/1xfVPDuOAA3rc_6_1JaR5QQ?pwd=673u) | 


**说明：** 
1. 这里字错率或者词错率是使用`eval.py`程序并使用集束搜索解码`ctc_beam_search`方法计算得到的。
2. 把全部文件复制到项目根目录下。
3. 模型名称包含`no_stream`为非流式模型，不能用于流式识别。
4. 由于算力不足，大部分的模型都没有训练足够轮数，具体轮数可以查看`model.state`文件。
5. 由于音频的长度不一，所以参数大小也有所变化，以上参数大小为同一音频长度下的结果，仅供对比使用。

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
   - [本地预测](./docs/infer.md)
   - [长语音预测](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)
   - [GUI界面预测](./docs/infer.md)
   - [Nvidia Jetson部署](./docs/nvidia-jetson.md)


## 快速预测

 - 下载作者提供的模型，或者训练模型并[导出模型](./docs/export_model.md)，使用`infer_path.py`预测音频，通过参数`--wav_path`指定需要预测的音频路径，完成语音识别，详情请查看[模型部署](./docs/infer.md)。
```shell script
python infer_path.py --wav_path=./dataset/test.wav
```

输出结果：
```
----------- 额外配置参数 -----------
configs: configs/config_zh.yml
is_long_audio: False
model_dir: models/{}_{}/infer/
pun_model_dir: models/pun_models/
real_time_demo: False
to_an: False
use_gpu: True
use_pun: False
wav_path: dataset/test.wav
------------------------------------------------
----------- 配置文件参数 -----------
ctc_beam_search_decoder: {'alpha': 2.2, 'beta': 4.3, 'beam_size': 300, 'num_processes': 10, 'cutoff_prob': 0.99, 'cutoff_top_n': 40, 'language_model_path': 'lm/zh_giga.no_cna_cmn.prune01244.klm'}
dataset: {'batch_size': 32, 'num_workers': 4, 'min_duration': 0.5, 'max_duration': 20, 'train_manifest': 'dataset/manifest.train', 'test_manifest': 'dataset/manifest.test', 'dataset_vocab': 'dataset/vocabulary.txt', 'mean_std_path': 'dataset/mean_std.json', 'noise_manifest_path': 'dataset/manifest.noise'}
decoder: ctc_beam_search
metrics_type: cer
num_epoch: 65
optimizer: {'learning_rate': '5e-5', 'gamma': 0.93, 'clip_norm': 3.0, 'weight_decay': '1e-6'}
preprocess: {'feature_method': 'fbank', 'n_mels': 80, 'n_mfcc': 40, 'sample_rate': 16000, 'use_dB_normalization': True, 'target_dB': -20}
use_model: deepspeech2
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
