# PPASR语音识别（进阶级）

本项目将分三个阶段分支，分别是入门级、进阶级和最终级分支，当前为进阶级，随着级别的提升，识别准确率也随之提升，也更适合实际项目使用，敬请关注！

PPASR基于PaddlePaddle2实现的端到端自动语音识别，相比入门级，进阶级从三个方面来提高模型的准确率，首先最主要的是更换了模型，这次采用了DeepSpeech2模型，DeepSpeech2是2015年百度发布的语音识别模型，其论文为[《Baidu’s Deep Speech 2 paper》](http://proceedings.mlr.press/v48/amodei16.pdf) 。然后也修改了音频的预处理，这次使用了在语音识别上更好的预处理，通过用FFT energy计算线性谱图。最好修改的是解码器，相比之前使用的贪心策略解码器，这次增加了波束搜索解码器，这个解码器可以加载语言模型，对解码的结果调整，使得预测输出语句更合理，从而提高准确率。

# 安装环境

 - 本项目的训练在Windows或者Ubuntu都可以运行，安装环境很简单，只需要执行以下一条命令即可。
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

 - 在评估和预测都可以选择不同的解码器，如果是选择波束搜索解码器，就需要执行下面命令来安装环境，该解码器只支持Linux编译安装。如果使用的是Windows，那么就只能选择贪心策略解码器。
```shell
cd decoders
sh setup.sh
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

 - 执行下面的命令，创建数据列表，以及建立词表，也就是数据字典，把所有出现的字符都存放子在`vocabulary.json`文件中，生成的文件都存放在`dataset/`目录下。在图像预处理的时候需要用到均值和标准值，之后的评估和预测同样需要用到，这些都会计算并保存在文件中。
```shell script
python3 create_manifest.py
```

我们来说说这些文件和数据的具体作用，创建数据列表是为了在训练是读取数据，读取数据程序通过读取图像列表的每一行都能得到音频的文件路径、音频长度以及这句话的内容。通过路径读取音频文件并进行预处理，音频长度用于统计数据总长度，文字内容就是输入数据的标签，在训练是还需要数据字典把这些文字内容转置整型的数字，比如`是`这个字在数据字典中排在第5，那么它的标签就是4，标签从0开始。至于最后生成的均值和标准值，因为我们的数据在训练之前还需要归一化，因为每个数据的分布不一样，不同图像，最大最小值都是确定的，随机采取一部分的书籍计算均值和标准值，然后把均值和标准值保存在`npy`文件中。

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

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`models/`目录下，测试使用的是贪心解码路径解码方法。本项目支持多卡训练，通过使用`--gpus`参数指定，如`--gpus=0,1`指定使用第1张和第2张显卡训练。其他的参数一般不需要改动，参数`--num_workers`可以数据读取的线程数，这个参数是指定使用多少个线程读取数据。参数`--pretrained_model`是指定预训练模型所在的文件夹，使用预训练模型，在加载的时候会自动跳过维度不一致的层。如果使用`--resume`恢复训练模型，恢复模型的路径结构应该要跟保存的时候一样，这样才能读取到该模型是epoch数，并且必须使用跟预训练配套的数据字典，原因是，其一，数据字典的大小指定了模型的输出大小，如果使用了其他更大的数据字典，恢复训练模型就无法完全加载。其二，数值字典定义了文字的ID，不同的数据字典文字的ID可能不一样，这样预训练模型的作用就不是那么大了。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python3 train.py
```

训练输出结果如下：
```shell
-----------  Configuration Arguments -----------
batch_size: 32
data_mean: -3.146301
data_std: 52.998405
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
input_size的第三个参数是变长的，这里为了能查看输出的大小变化，指定了一个值！
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Conv1D-1       [[32, 128, 500]]      [32, 500, 324]       3,073,000   
   Sigmoid-1      [[32, 250, 324]]      [32, 250, 324]           0       
     GLU-1        [[32, 500, 324]]      [32, 250, 324]           0       
   Dropout-1      [[32, 250, 324]]      [32, 250, 324]           0       
  ConvBlock-1     [[32, 128, 500]]      [32, 250, 324]           0       
   Conv1D-2       [[32, 250, 288]]      [32, 500, 282]        876,000    
   Sigmoid-2      [[32, 250, 282]]      [32, 250, 282]           0       
     GLU-2        [[32, 500, 282]]      [32, 250, 282]           0       
   Dropout-2      [[32, 250, 282]]      [32, 250, 282]           0       
  ConvBlock-2     [[32, 250, 288]]      [32, 250, 282]           0       
   Conv1D-3       [[32, 250, 282]]     [32, 2000, 251]      16,004,000   
   Sigmoid-3     [[32, 1000, 251]]     [32, 1000, 251]           0       
     GLU-3       [[32, 2000, 251]]     [32, 1000, 251]           0       
   Dropout-3     [[32, 1000, 251]]     [32, 1000, 251]           0       
  ConvBlock-3     [[32, 250, 282]]     [32, 1000, 251]           0       
   Conv1D-4      [[32, 1000, 251]]     [32, 2000, 251]       2,004,000   
   Sigmoid-4     [[32, 1000, 251]]     [32, 1000, 251]           0       
     GLU-4       [[32, 2000, 251]]     [32, 1000, 251]           0       
   Dropout-4     [[32, 1000, 251]]     [32, 1000, 251]           0       
  ConvBlock-4    [[32, 1000, 251]]     [32, 1000, 251]           0       
   Conv1D-5      [[32, 1000, 251]]     [32, 4323, 251]       4,331,646   
===========================================================================
Total params: 26,288,646
Trainable params: 26,288,646
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 7.81
Forward/backward pass size (MB): 1222.19
Params size (MB): 100.28
Estimated Total Size (MB): 1330.28
---------------------------------------------------------------------------
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
                        数据集的均值 默认: -3.146301.
  --data_std DATA_STD   数据集的标准值 默认: 52.998405.
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

![Train](https://s3.ax1x.com/2021/03/05/6ehhjK.jpg)


# 评估和预测

在评估和预测中，对结果解码的贪心策略解码方法，贪心策略是在每一步选择概率最大的输出值，这样就可以得到最终解码的输出序列。然而，CTC网络的输出序列只对应了搜索空间的一条路径，一个最终标签可对应搜索空间的N条路径，所以概率最大的路径并不等于最终标签的概率最大，即不是最优解。但贪心策略是最简单易懂且快速地一种方法。在语音识别上使用最多的解码方法还有定向搜索策略，这种策略准确率更高，同时也相对复杂，解码速度也相对慢很多。

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。目前只支持贪心策略解码方法。在评估中音频预处理的`mean`和`std`需要跟训练时一样，但这里不需要开发者手动指定，因为这两个参数在训练的时候就已经保持在模型中，这时只需从模型中读取这两个参数的值就可以。参数`model_path`指定模型所在的文件夹的路径。
```shell script
python3 eval.py --model_path=models/step_final/
```

可以用使用`python eval.py --help`命令查看各个参数的说明和默认值。
```shell
usage: eval.py [-h] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
               [--test_manifest TEST_MANIFEST] [--dataset_vocab DATASET_VOCAB]
               [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        训练的批量大小 默认: 32.
  --num_workers NUM_WORKERS
                        读取数据的线程数量 默认: 8.
  --test_manifest TEST_MANIFEST
                        测试数据的数据列表路径 默认: dataset/manifest.test.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/zh_vocab.json.
  --model_path MODEL_PATH
                        模型的路径 默认: models/step_final/.
```

 - 我们可以使用这个脚本使用模型进行预测，通过传递音频文件的路径进行识别。在预测中音频预处理的`mean`和`std`需要跟训练时一样，但这里不需要开发者手动指定，因为这两个参数在训练的时候就已经保持在模型中，这时只需从模型中读取这两个参数的值就可以。参数`model_path`指定模型所在的文件夹的路径，参数`wav_path`指定需要预测音频文件的路径。
```shell script
python3 infer.py --audio_path=./dataset/test.wav
```

可以用使用`python infer.py --help`命令查看各个参数的说明和默认值。
```shell
usage: infer.py [-h] [--audio_path AUDIO_PATH] [--dataset_vocab DATASET_VOCAB]
                [--model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --audio_path AUDIO_PATH
                        用于识别的音频路径 默认: dataset/test.wav.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/zh_vocab.json.
  --model_path MODEL_PATH
                        模型的路径 默认: models/step_final/.
```

## 模型下载
| 数据集 | 字错率 | 下载地址 |
| :---: | :---: | :---: |
| AISHELL | 0.08 | [点击下载]() |
