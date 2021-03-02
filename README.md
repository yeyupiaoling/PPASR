# PPASR

PPASR基于PaddlePaddle2实现的端到端自动语音识别，本项目最大的特点简单，在保证准确率不低的情况下，项目尽量做得浅显易懂，能够让每个想入门语音识别的开发者都能够轻松上手。PPASR只使用卷积神经网络，无其他特殊网络结构，模型简单易懂。

# 数据准备

1. 在`data`目录下是公开数据集的下载和制作训练数据列表和字典的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如果想快速训练，也可以只下载其中一个。
```shell script
cd data/
python3 aishell.py
python3 free_st_chinese_mandarin_corpus.py
python3 thchs_30.py
```

 - 如果开发者有自己的数据集，可以使用自己的数据集进行训练，当然也可以跟上面下载的数据集一起训练。自定义的语音数据需要符合一下格式：
    1. 语音文件需要放在`dataset/audio/`目录下，例如我们有个`wav`的文件夹，里面都是语音文件，我们就把这个文件存放在`dataset/audio/`。
    2. 然后把数据列表文件存在`dataset/annotation/`目录下，程序会遍历这个文件下的所有数据列表文件。例如这个文件下存放一个`my_audio.txt`，它的内容格式如下。每一行数据包含该语音文件的相对路径和该语音文件对应的中文文本，要注意的是该中文文本只能包含纯中文，不能包含标点符号、阿拉伯数字以及英文字母。
```shell script
dataset/audio/wav/0175/H0175A0171.wav 我需要把空调温度调到二十度
dataset/audio/wav/0175/H0175A0377.wav 出彩中国人
dataset/audio/wav/0175/H0175A0470.wav 据克而瑞研究中心监测
dataset/audio/wav/0175/H0175A0180.wav 把温度加大到十八
```

 - 执行下面的命令，创建数据列表，以及建立词表，也就是数据字典，把所有出现的字符都存放子在`zh_vocab.json`文件中，生成的文件都存放在`dataset/`目录下。
```shell script
python3 create_manifest.py
```


# 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`models/`目录下，测试使用的是最优解码路径解码方法。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python3 train.py
```

训练输出结果如下：
```shell
-----------  Configuration Arguments -----------
batch_size: 32
dataset_vocab: dataset/zh_vocab.json
learning_rate: 0.001
num_epoch: 200
num_workers: 8
pretrained_model: None
save_model: models/
test_manifest: dataset/manifest.test
train_manifest: dataset/manifest.train
------------------------------------------------
I0302 09:24:11.613675 15693 nccl_context.cc:189] init nccl context nranks: 2 local rank: 1 gpu id: 1 ring id: 0
I0302 09:24:11.613677 15692 nccl_context.cc:189] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
W0302 09:24:11.791157 15692 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0302 09:24:11.791216 15693 device_context.cc:362] Please NOTE: device: 1, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0302 09:24:11.793020 15693 device_context.cc:372] device: 1, cuDNN Version: 7.6.
W0302 09:24:11.793021 15692 device_context.cc:372] device: 0, cuDNN Version: 7.6.
epoch 0, batch 0, loss: 155.009735
epoch 0, batch 100, loss: 9.400544
epoch 0, batch 200, loss: 8.898541
epoch 0, batch 300, loss: 8.006586
epoch 0, batch 400, loss: 8.478329
epoch 0, batch 500, loss: 7.810921
epoch 0, batch 600, loss: 7.690178
epoch 0, batch 700, loss: 7.705078
epoch 0, batch 800, loss: 7.515574
epoch 0, batch 900, loss: 7.738822
```

 - 在训练过程中，程序会使用VisualDL记录训练结果，可以通过以下的命令启动VisualDL。
```shell
visualdl --logdir=log --host 0.0.0.0
```

 - 然后再浏览器上访问`http://localhost:8040`可以查看结果显示，如下。

![Test Cer](https://s3.ax1x.com/2021/03/01/6PJaZV.jpg)
![Train Loss](https://s3.ax1x.com/2021/03/01/6PJNq0.jpg)


# 评估和预测

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。目前只支持最优解码路径解码方法。
```shell script
python3 eval.py --model_path=models/step_final/
```

 - 我们可以使用这个脚本使用模型进行预测，通过传递音频文件的路径进行识别。
```shell script
python3 infer_path.py --wav_path=./dataset/test.wav
```
