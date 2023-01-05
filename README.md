![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/PPASR)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/PPASR)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/PPASR)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# PPASR流式与非流式语音识别项目

本项目将分三个阶段分支，分别是[入门级](https://github.com/yeyupiaoling/PPASR/tree/%E5%85%A5%E9%97%A8%E7%BA%A7) 、[进阶级](https://github.com/yeyupiaoling/PPASR/tree/%E8%BF%9B%E9%98%B6%E7%BA%A7) 和[最终级](https://github.com/yeyupiaoling/PPASR) 分支，当前为最终级的V2版本，如果想使用最终级的V1版本，请在这个分支[r1.x](https://github.com/yeyupiaoling/PPASR/tree/r1.x)。PPASR中文名称PaddlePaddle中文语音识别（PaddlePaddle Automatic Speech Recognition），是一款基于PaddlePaddle实现的语音识别框架，PPASR致力于简单，实用的语音识别项目。可部署在服务器，Nvidia Jetson设备，未来还计划支持Android等移动设备。**别忘了star**

**欢迎大家扫码入QQ群讨论**，或者直接搜索QQ群号`1169600237`，问题答案为博主Github的ID`yeyupiaoling`。

<div align="center">
  <img src="docs/images/qq.png"/>
</div>

## 在线使用

**1. [在AI Studio平台训练预测](https://aistudio.baidu.com/aistudio/projectdetail/3290199)**

**2. [在线使用Dome](http://ppasr.yeyupiaoling.cn:8081)**

<br/>

**本项目使用的环境：**
 - Anaconda 3
 - Python 3.8
 - PaddlePaddle 2.4.1
 - Windows 10 or Ubuntu 18.04


## 项目快速了解

 1. 本项目支持流式识别模型`deepspeech2`、`conformer`、`squeezeformer`，每个模型又分online(在线)和offline(离线)，对应的是流式识别和非流式识别。
 2. 本项目支持两种解码器，分别是集束搜索解码器`ctc_beam_search`和贪心解码器`ctc_greedy`，集束搜索解码器`ctc_beam_search`准确率更高，但不支持Windows。
 3. 下面提供了一系列预训练模型的下载，下载预训练模型之后，需要把全部文件复制到项目根目录，并执行导出模型才可以使用语音识别。

## 更新记录

 - 2022.12.05: 支持自动混合精度训练和导出量化模型。
 - 2022.11.26: 支持Squeezeformer模型。
 - 2022.11.01: 修改Conformer模型的解码器为BiTransformerDecoder，增加SpecSubAugmentor数据增强器。
 - 2022.10.29: 正式发布最终级的V2版本。

## 视频讲解

 - [知识点讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Rr4y1D7iZ)
 - [流式识别的使用讲解（哔哩哔哩）](https://www.bilibili.com/video/BV1Te4y1h7KK)


# 快速使用

这里介绍如何使用PPASR快速进行语音识别，前提是要安装PPASR，文档请看[快速安装](./docs/install.md)。执行过程不需要手动下载模型，全部自动完成。

1. 短语音识别
```python
from ppasr.predict import PPASRPredictor

predictor = PPASRPredictor(model_tag='conformer_online_fbank_wenetspeech')

wav_path = 'dataset/test.wav'
result = predictor.predict(audio_data=wav_path, use_pun=False)
score, text = result['score'], result['text']
print(f"识别结果: {text}, 得分: {int(score)}")
```

2. 长语音识别
```python
from ppasr.predict import PPASRPredictor

predictor = PPASRPredictor(model_tag='conformer_online_fbank_wenetspeech')

wav_path = 'dataset/test_long.wav'
result = predictor.predict_long(audio_data=wav_path, use_pun=False)
score, text = result['score'], result['text']
print(f"识别结果: {text}, 得分: {score}")
```

3. 模拟流式识别
```python
import time
import wave

from ppasr.predict import PPASRPredictor

predictor = PPASRPredictor(model_tag='conformer_online_fbank_wenetspeech')

# 识别间隔时间
interval_time = 0.5
CHUNK = int(16000 * interval_time)
# 读取数据
wav_path = 'dataset/test.wav'
wf = wave.open(wav_path, 'rb')
data = wf.readframes(CHUNK)
# 播放
while data != b'':
    start = time.time()
    d = wf.readframes(CHUNK)
    result = predictor.predict_stream(audio_data=data, use_pun=False, is_end=d == b'')
    data = d
    if result is None: continue
    score, text = result['score'], result['text']
    print(f"【实时结果】：消耗时间：{int((time.time() - start) * 1000)}ms, 识别结果: {text}, 得分: {int(score)}")
# 重置流式识别
predictor.reset_stream()
```


## 模型下载

1. `conformer`预训练模型列表：

|       使用模型        |                                                             数据集                                                             | 预处理方式 | 语言  |                             测试集字错率（词错率）                             |                               下载地址                               |
|:-----------------:|:---------------------------------------------------------------------------------------------------------------------------:|:-----:|:---:|:-------------------------------------------------------------------:|:----------------------------------------------------------------:|
| conformer_online  |                                       [WenetSpeech](./docs/wenetspeech.md) (10000小时)                                        | fbank | 中文  | 0.03579(aishell_test)<br>0.11081(test_net)<br>0.16031(test_meeting) | [点击下载](https://download.csdn.net/download/qq_33200967/86932770)  |
| conformer_online  | [WenetSpeech](./docs/wenetspeech.md) (10000小时)+[中文语音数据集](https://download.csdn.net/download/qq_33200967/87003964) (3000+小时) | fbank | 中文  | 0.02923(aishell_test)<br>0.11876(test_net)<br>0.18346(test_meeting) | [点击下载](https://download.csdn.net/download/qq_33200967/86951249)  |
| conformer_online  |                              [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)                              | fbank | 中文  |                               0.04936                               | [点击下载](https://pan.baidu.com/s/1LI29m53S1-x_BPsLV4S87A?pwd=9f0f) |
| conformer_offline |                              [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)                              | fbank | 中文  |                               0.04343                               | [点击下载](https://pan.baidu.com/s/1LI29m53S1-x_BPsLV4S87A?pwd=9f0f) |
| conformer_online  |                            [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时)                            | fbank | 英文  |                               0.08109                               | [点击下载](https://pan.baidu.com/s/1LNMwj7YsUUIzagegivsw8A?pwd=ly84) | 
| conformer_offline |                            [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时)                            | fbank | 英文  |                               0.08036                               | [点击下载](https://pan.baidu.com/s/1LNMwj7YsUUIzagegivsw8A?pwd=ly84) | 


2. `squeezeformer`预训练模型列表：

|         使用模型          |                                  数据集                                  | 预处理方式 | 语言  | 测试集字错率（词错率） |                               下载地址                               |
|:---------------------:|:---------------------------------------------------------------------:|:-----:|:---:|:-----------:|:----------------------------------------------------------------:|
| squeezeformer_online  |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |   0.04927   | [点击下载](https://pan.baidu.com/s/1LI29m53S1-x_BPsLV4S87A?pwd=9f0f) |
| squeezeformer_offline |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |   0.04889   | [点击下载](https://pan.baidu.com/s/1LI29m53S1-x_BPsLV4S87A?pwd=9f0f) |
| squeezeformer_online  | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |             | [点击下载](https://pan.baidu.com/s/1LNMwj7YsUUIzagegivsw8A?pwd=ly84) | 
| squeezeformer_offline | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |             | [点击下载](https://pan.baidu.com/s/1LNMwj7YsUUIzagegivsw8A?pwd=ly84) | 



3. `deepspeech2`预训练模型列表：

|        使用模型         |                                  数据集                                  | 预处理方式 | 语言  |      测试集字错率（词错率）      |                               下载地址                               |
|:-------------------:|:---------------------------------------------------------------------:|:-----:|:---:|:---------------------:|:----------------------------------------------------------------:|
| deepspeech2_online  |            [WenetSpeech](./docs/wenetspeech.md) (10000小时)             | fbank | 中文  | 0.05379(aishell_test) | [点击下载](https://download.csdn.net/download/qq_33200967/86932787)  |
| deepspeech2_online  |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |        0.11367        | [点击下载](https://pan.baidu.com/s/1LI29m53S1-x_BPsLV4S87A?pwd=9f0f) |
| deepspeech2_offline |   [aishell](https://openslr.magicdatatech.com/resources/33) (179小时)   | fbank | 中文  |        0.09385        | [点击下载](https://pan.baidu.com/s/1LI29m53S1-x_BPsLV4S87A?pwd=9f0f) |
| deepspeech2_online  | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |        0.15294        | [点击下载](https://pan.baidu.com/s/1LNMwj7YsUUIzagegivsw8A?pwd=ly84) | 
| deepspeech2_offline | [Librispeech](https://openslr.magicdatatech.com/resources/12) (960小时) | fbank | 英文  |        0.11035        | [点击下载](https://pan.baidu.com/s/1LNMwj7YsUUIzagegivsw8A?pwd=ly84) | 


**说明：** 
1. 这里字错率或者词错率是使用`eval.py`程序并使用集束搜索解码`ctc_beam_search`方法计算得到的，`min_duration`为1.0，`max_duration`为20.0。
2. 没有提供预测模型，需要把全部文件复制到项目的根目录下，执行`export_model.py`导出预测模型。

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
- [使用语音活动检测（VAD）](./docs/vad.md)
- 预测
   - [本地预测](./docs/infer.md)
   - [长语音预测](./docs/infer.md)
   - [Web部署模型](./docs/infer.md)
   - [GUI界面预测](./docs/infer.md)
   - [Nvidia Jetson部署](./docs/nvidia-jetson.md)

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
