# PPASR

PPASR基于PaddlePaddle2实现的端到端自动语音识别，本项目最大的特点简单，在保证准确率不低的情况下，项目尽量做得浅显易懂，能够让每个想入门语音识别的开发者都能够轻松上手。PPASR只使用卷积神经网络，无其他特殊网络结构，模型简单易懂。在数据预处理方便，本项目主要是将音频执行梅尔频率倒谱系数(MFCCs)处理，然后在使用出来的数据进行训练，在读取音频时，使用`wave.open()`库读取文件，再使用`librosa.feature.mfcc()`执行数据处理，这样可以加快数据的读取，因为使用`librosa.load()`读取音频文件比较慢。本项目使用的全部音频的采样率都是16000Hz，如果其他采样率的音频都需要转为16000Hz，`create_manifest.py`程序也提供了把音频转为16000Hz。


# 安装环境

 - 本项目可以在Windows或者Ubuntu都可以运行，安装环境很简单，只需要执行以下一条命令即可。
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```


# 数据准备

1. 在`data`目录下是公开数据集的下载和制作训练数据列表和字典的，本项目提供了下载公开的中文普通话语音数据集，分别是Aishell，Free ST-Chinese-Mandarin-Corpus，THCHS-30 这三个数据集，总大小超过28G。下载这三个数据只需要执行一下代码即可，当然如果想快速训练，也可以只下载其中一个。
```shell script
python3 data/aishell.py
python3 data/free_st_chinese_mandarin_corpus.py
python3 data/thchs_30.py
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

 - 执行下面的命令，创建数据列表，以及建立词表，也就是数据字典，把所有出现的字符都存放子在`zh_vocab.json`文件中，生成的文件都存放在`dataset/`目录下。最最最重要的是还计算了数据集的均值和标准值，计算得到的均值和标准值需要更新在训练参数`data_mean`和`data_std`中，之后的评估和预测同样需要用到。有几个参数需要注意，参数`is_change_frame_rate`是指定在生成数据集的时候，是否要把音频的采样率转换为16000Hz，最好是使用默认值。参数`min_duration`和`max_duration`限制音频的长度，特别是有些音频太长，会导致显存不足，训练直接崩掉。
```shell script
python3 create_manifest.py
```

我们来说说这些文件和数据的具体作用，创建数据列表是为了在训练是读取数据，读取数据程序通过读取图像列表的每一行都能得到音频的文件路径、音频长度以及这句话的内容。通过路径读取音频文件并进行预处理，音频长度用于统计数据总长度，文字内容就是输入数据的标签，在训练是还需要数据字典把这些文字内容转置整型的数字，比如`是`这个字在数据字典中排在第5，那么它的标签就是4，标签从0开始。至于最后生成的均值和标准值，因为我们的数据在训练之前还需要归一化，因为每个数据的分布不一样，不同图像，最大最小值都是确定的，所以我们要统计一批数据来计算均值和标准值，之后的数据的归一化都使用这个均值和标准值。

输出结果如下：
```shell
-----------  Configuration Arguments -----------
annotation_path: dataset/annotation/
count_threshold: 0
is_change_frame_rate: True
manifest_path: dataset/manifest.train
manifest_prefix: dataset/
max_duration: 20
min_duration: 0
vocab_path: dataset/zh_vocab.json
------------------------------------------------
开始生成数据列表...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 141600/141600 [00:17<00:00, 8321.22it/s]
完成生成数据列表，数据集总长度为178.97小时！
开始生成数据字典...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 140184/140184 [00:01<00:00, 89476.12it/s]
数据字典生成完成！
开始抽取1%的数据计算均值和标准值...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 140184/140184 [01:33<00:00, 1507.15it/s]
【特别重要】：均值：1.424120, 标准值：0.947314, 请根据这两个值修改训练参数！
```

可以用使用`python create_manifest.py --help`命令查看各个参数的说明和默认值。
```shell
usage: create_manifest.py [-h] [----annotation_path ANNOTATION_PATH]
                          [--manifest_prefix MANIFEST_PREFIX]
                          [--is_change_frame_rate IS_CHANGE_FRAME_RATE]
                          [--min_duration MIN_DURATION]
                          [--max_duration MAX_DURATION]
                          [--count_threshold COUNT_THRESHOLD]
                          [--vocab_path VOCAB_PATH]
                          [--manifest_path MANIFEST_PATH]

optional arguments:
  -h, --help            show this help message and exit
  ----annotation_path ANNOTATION_PATH
                        标注文件的路径 默认: dataset/annotation/.
  --manifest_prefix MANIFEST_PREFIX
                        训练数据清单，包括音频路径和标注信息 默认: dataset/.
  --is_change_frame_rate IS_CHANGE_FRAME_RATE
                        是否统一改变音频为16000Hz，这会消耗大量的时间 默认: True.
  --min_duration MIN_DURATION
                        过滤最短的音频长度 默认: 0.
  --max_duration MAX_DURATION
                        过滤最长的音频长度，当为-1的时候不限制长度 默认: 20.
  --count_threshold COUNT_THRESHOLD
                        字符计数的截断阈值，0为不做限制 默认: 0.
  --vocab_path VOCAB_PATH
                        生成的数据字典文件 默认: dataset/zh_vocab.json.
  --manifest_path MANIFEST_PATH
                        数据列表路径 默认: dataset/manifest.train.
```

# 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`models/`目录下，测试使用的是最优解码路径解码方法。本项目支持多卡训练，在没有指定`CUDA_VISIBLE_DEVICES`时，会使用全部的GPU进行执行训练，也可以指定某几个GPU训练，如`CUDA_VISIBLE_DEVICES=0,1`指定使用第1张和第2张显卡训练。除了参数`data_mean`和`data_std`需要根据计算的结果修改，其他的参数一般不需要改动，参数`num_workers`可以更加CPU的核数修改，这个参数是指定使用多少个线程读取数据。参数`pretrained_model`是指定预训练模型所在的文件夹，如果使用训练模型，必须使用跟预训练配套的数据字典，原因是，其一，数据字典的大小指定了模型的输出大小，如果使用了其他更大的数据字典，预训练模型就无法完全加载。其二，数值字典定义了文字的ID，不同的数据字典文字的ID可能不一样，这样预训练模型的作用就不是那么大了。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python3 train.py
```

训练输出结果如下：
```shell
-----------  Configuration Arguments -----------
batch_size: 32
data_mean: 1.424366
data_std: 0.944142
dataset_vocab: dataset/zh_vocab.json
learning_rate: 0.001
num_epoch: 200
num_workers: 8
pretrained_model: None
save_model: models/
test_manifest: dataset/manifest.test
train_manifest: dataset/manifest.train
------------------------------------------------
I0303 16:55:39.645823 16572 nccl_context.cc:189] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
I0303 16:55:39.645821 16573 nccl_context.cc:189] init nccl context nranks: 2 local rank: 1 gpu id: 1 ring id: 0
W0303 16:55:39.905000 16572 device_context.cc:362] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0303 16:55:39.905090 16573 device_context.cc:362] Please NOTE: device: 1, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0303 16:55:39.907197 16572 device_context.cc:372] device: 0, cuDNN Version: 7.6.
W0303 16:55:39.907199 16573 device_context.cc:372] device: 1, cuDNN Version: 7.6.
Epoch 0: ExponentialDecay set learning rate to 0.001.
Epoch 0: ExponentialDecay set learning rate to 0.001.
[2021-03-03 16:56:01.754491] Train epoch 0, batch 0, loss: 269.343811
[2021-03-03 16:58:08.436214] Train epoch 0, batch 100, loss: 7.195621
[2021-03-03 16:59:54.781490] Train epoch 0, batch 200, loss: 6.914866
[2021-03-03 17:01:34.841955] Train epoch 0, batch 300, loss: 6.824973
[2021-03-03 17:03:09.492905] Train epoch 0, batch 400, loss: 6.905243
```

可以用使用`python train.py --help`命令查看各个参数的说明和默认值。
```shell
usage: train.py [-h] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--num_epoch NUM_EPOCH] [--learning_rate LEARNING_RATE]
                [--data_mean DATA_MEAN] [--data_std DATA_STD]
                [--train_manifest TRAIN_MANIFEST]
                [--test_manifest TEST_MANIFEST]
                [--dataset_vocab DATASET_VOCAB] [--save_model SAVE_MODEL]
                [--pretrained_model PRETRAINED_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        训练的批量大小 默认: 32.
  --num_workers NUM_WORKERS
                        读取数据的线程数量 默认: 8.
  --num_epoch NUM_EPOCH
                        训练的轮数 默认: 200.
  --learning_rate LEARNING_RATE
                        初始学习率的大小 默认: 0.001.
  --data_mean DATA_MEAN
                        数据集的均值 默认: 1.424366.
  --data_std DATA_STD   数据集的标准值 默认: 0.944142.
  --train_manifest TRAIN_MANIFEST
                        训练数据的数据列表路径 默认: dataset/manifest.train.
  --test_manifest TEST_MANIFEST
                        测试数据的数据列表路径 默认: dataset/manifest.test.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/zh_vocab.json.
  --save_model SAVE_MODEL
                        模型保存的路径 默认: models/.
  --pretrained_model PRETRAINED_MODEL
                        预训练模型的路径，当为None则不使用预训练模型 默认: None.
```

 - 在训练过程中，程序会使用VisualDL记录训练结果，可以通过以下的命令启动VisualDL。
```shell
visualdl --logdir=log --host 0.0.0.0
```

 - 然后再浏览器上访问`http://localhost:8040`可以查看结果显示，如下。

![Test Cer](https://s3.ax1x.com/2021/03/01/6PJaZV.jpg)
![Train Loss](https://s3.ax1x.com/2021/03/01/6PJNq0.jpg)


# 评估和预测

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。目前只支持最优解码路径解码方法。参数`data_mean`和`data_std`需要跟训练时一样，参数`model_path`指定模型所在的文件夹的路径。
```shell script
python3 eval.py --model_path=models/step_final/
```

可以用使用`python eval.py --help`命令查看各个参数的说明和默认值。
```shell
usage: eval.py [-h] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
               [--data_mean DATA_MEAN] [--data_std DATA_STD]
               [--test_manifest TEST_MANIFEST] [--dataset_vocab DATASET_VOCAB]
               [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        训练的批量大小 默认: 32.
  --num_workers NUM_WORKERS
                        读取数据的线程数量 默认: 8.
  --data_mean DATA_MEAN
                        数据集的均值 默认: 1.424366.
  --data_std DATA_STD   数据集的标准值 默认: 0.944142.
  --test_manifest TEST_MANIFEST
                        测试数据的数据列表路径 默认: dataset/manifest.test.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/zh_vocab.json.
  --model_path MODEL_PATH
                        模型的路径 默认: models/step_final/.
```

 - 我们可以使用这个脚本使用模型进行预测，通过传递音频文件的路径进行识别。参数`data_mean`和`data_std`需要跟训练时一样，参数`model_path`指定模型所在的文件夹的路径，参数`wav_path`指定需要预测音频文件的路径，注意这个音频的采样率必须是16000Hz，如果不是，需要要转换在执行预测。
```shell script
python3 infer.py --wav_path=./dataset/test.wav
```
<audio controls>
  <source src="/dataset/test.wav" type="audio/wav">
</audio>

可以用使用`python infer.py --help`命令查看各个参数的说明和默认值。
```shell
usage: infer.py [-h] [--audio_path AUDIO_PATH] [--data_mean DATA_MEAN]
                [--data_std DATA_STD] [--dataset_vocab DATASET_VOCAB]
                [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --audio_path AUDIO_PATH
                        用于识别的音频路径 默认: dataset/test.wav.
  --data_mean DATA_MEAN
                        数据集的均值 默认: 1.424366.
  --data_std DATA_STD   数据集的标准值 默认: 0.944142.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/zh_vocab.json.
  --model_path MODEL_PATH
                        模型的路径 默认: models/step_final/.
```
