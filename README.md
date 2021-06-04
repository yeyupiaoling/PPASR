# PPASR语音识别（进阶级）

本项目将分三个阶段分支，分别是[入门级](https://github.com/yeyupiaoling/PPASR/tree/%E5%85%A5%E9%97%A8%E7%BA%A7) 、[进阶级](https://github.com/yeyupiaoling/PPASR/tree/%E8%BF%9B%E9%98%B6%E7%BA%A7) 和[最终级](https://github.com/yeyupiaoling/PPASR) 分支，当前为进阶级，随着级别的提升，识别准确率也随之提升，也更适合实际项目使用，敬请关注！

PPASR基于PaddlePaddle2实现的端到端自动语音识别，相比入门级，进阶级从三个方面来提高模型的准确率，首先最主要的是更换了模型，这次采用了DeepSpeech2模型，DeepSpeech2是2015年百度发布的语音识别模型，其论文为[《Baidu’s Deep Speech 2 paper》](http://proceedings.mlr.press/v48/amodei16.pdf) 。然后也修改了音频的预处理，这次使用了在语音识别上更好的预处理，通过用FFT energy计算线性谱图。最好修改的是解码器，相比之前使用的贪心策略解码器，这次增加了波束搜索解码器，这个解码器可以加载语言模型，对解码的结果调整，使得预测输出语句更合理，从而提高准确率。

# 安装环境

 - 本项目的训练在Windows或者Ubuntu都可以运行，安装环境很简单，只需要执行以下一条命令即可。
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**如果出现LLVM版本错误**，则执行下面的命令，然后重新执行上面的安装命令，否则不需要执行。
```shell
cd ~
wget https://releases.llvm.org/9.0.0/llvm-9.0.0.src.tar.xz
wget http://releases.llvm.org/9.0.0/cfe-9.0.0.src.tar.xz
wget http://releases.llvm.org/9.0.0/clang-tools-extra-9.0.0.src.tar.xz
tar xvf llvm-9.0.0.src.tar.xz
tar xvf cfe-9.0.0.src.tar.xz
tar xvf clang-tools-extra-9.0.0.src.tar.xz
mv llvm-9.0.0.src llvm-src
mv cfe-9.0.0.src llvm-src/tools/clang
mv clang-tools-extra-9.0.0.src llvm-src/tools/clang/tools/extra
sudo mkdir -p /usr/local/llvm
sudo mkdir -p llvm-src/build
cd llvm-src/build
sudo cmake -G "Unix Makefiles" -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX="/usr/local/llvm" ..
sudo make -j8
sudo make install
export LLVM_CONFIG=/usr/local/llvm/bin/llvm-config
```

 - 在评估和预测都可以选择不同的解码器，如果是选择波束搜索解码器，就需要执行下面命令来安装环境，该解码器只支持Linux编译安装。如果使用的是Windows，那么就只能选择贪心策略解码器，无需再执行下面的命令编译安装波束搜索解码器。
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
is_change_frame_rate: Ture
manifest_path: dataset/manifest.train
manifest_prefix: dataset/
num_samples: 5000
output_path: ./dataset/mean_std.npz
vocab_path: dataset/vocabulary.json
------------------------------------------------
开始生成数据列表...
100%|██████████| 141600/141600 [23:00<00:00, 102.57it/s] 
完成生成数据列表，数据集总长度为178.97小时！
开始生成数据字典...
100%|██████████| 140184/140184 [00:03<00:00, 43883.46it/s]
数据字典生成完成！
开始抽取5000条数据计算均值和标准值...
100%|██████████| 5000/5000 [01:53<00:00, 44.18it/s]
计算的均值和标准值已保存在 ./dataset/mean_std.npz！
```

可以用使用`python create_manifest.py --help`命令查看各个参数的说明和默认值。
```shell
optional arguments:
  -h, --help            show this help message and exit
  ----annotation_path ANNOTATION_PATH
                        标注文件的路径 默认: dataset/annotation/.
  --manifest_prefix MANIFEST_PREFIX
                        训练数据清单，包括音频路径和标注信息 默认: dataset/.
  --is_change_frame_rate IS_CHANGE_FRAME_RATE
                        是否统一改变音频为16000Hz，这会消耗大量的时间 默认: True.
  --count_threshold COUNT_THRESHOLD
                        字符计数的截断阈值，0为不做限制 默认: 0.
  --vocab_path VOCAB_PATH
                        生成的数据字典文件 默认: dataset/vocabulary.json.
  --manifest_path MANIFEST_PATH
                        数据列表路径 默认: dataset/manifest.train.
  --num_samples NUM_SAMPLES
                        用于计算均值和标准值得音频数量 默认: 5000.
  --output_path OUTPUT_PATH
                        保存均值和标准值得numpy文件路径，后缀 (.npz). 默认:
                        ./dataset/mean_std.npz.

```

# 训练模型

 - 执行训练脚本，开始训练语音识别模型， 每训练一轮保存一次模型，模型保存在`models/`目录下，测试使用的是贪心解码路径解码方法。本项目支持多卡训练，通过使用`--gpus`参数指定，如`--gpus=0,1`指定使用第1张和第2张显卡训练。其他的参数一般不需要改动，参数`--num_workers`可以数据读取的线程数，这个参数是指定使用多少个线程读取数据。参数`--pretrained_model`是指定预训练模型所在的文件夹，使用预训练模型，在加载的时候会自动跳过维度不一致的层。如果使用`--resume`恢复训练模型，恢复模型的路径结构应该要跟保存的时候一样，这样才能读取到该模型是epoch数，并且必须使用跟预训练配套的数据字典，原因是，其一，数据字典的大小指定了模型的输出大小，如果使用了其他更大的数据字典，恢复训练模型就无法完全加载。其二，数值字典定义了文字的ID，不同的数据字典文字的ID可能不一样，这样恢复练模型的作用就不是那么大了。
```shell script
CUDA_VISIBLE_DEVICES=0,1 python3 train.py
```

训练输出结果如下：
```shell
-----------  Configuration Arguments -----------
batch_size: 16
dataset_vocab: dataset/vocabulary.json
gpus: 0,1
learning_rate: 0.001
max_duration: 20
mean_std_path: dataset/mean_std.npz
min_duration: 0
num_conv_layers: 2
num_epoch: 50
num_rnn_layers: 3
num_workers: 8
pretrained_model: None
resume: None
rnn_layer_size: 1024
save_model: models/
test_manifest: dataset/manifest.test
train_manifest: dataset/manifest.train
------------------------------------------------
I0602 16:56:29.931150 26160 gen_comm_id_helper.cc:181] Server listening on: 127.0.0.1:44697 successful.
I0602 16:56:29.936925 26159 nccl_context.cc:74] init nccl context nranks: 2 local rank: 0 gpu id: 0 ring id: 0
I0602 16:56:29.936942 26160 nccl_context.cc:74] init nccl context nranks: 2 local rank: 1 gpu id: 1 ring id: 0
W0602 16:56:30.167621 26159 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0602 16:56:30.167665 26160 device_context.cc:404] Please NOTE: device: 1, GPU Compute Capability: 7.5, Driver API Version: 11.0, Runtime API Version: 10.2
W0602 16:56:30.169658 26160 device_context.cc:422] device: 1, cuDNN Version: 7.6.
W0602 16:56:30.169659 26159 device_context.cc:422] device: 0, cuDNN Version: 7.6.
input_size的第三个参数是变长的，这里为了能查看输出的大小变化，指定了一个值！
-------------------------------------------------------------------------------------
 Layer (type)         Input Shape              Output Shape             Param #    
=====================================================================================
   Conv2D-1       [[1, 1, 161, 970]]         [1, 32, 81, 324]           14,432     
 BatchNorm2D-1    [[1, 32, 81, 324]]         [1, 32, 81, 324]             128      
   ConvBn-1     [[1, 1, 161, 970], [1]]   [[1, 32, 81, 324], [1]]          0       
   Conv2D-2       [[1, 32, 81, 324]]         [1, 32, 41, 324]           236,544    
 BatchNorm2D-2    [[1, 32, 41, 324]]         [1, 32, 41, 324]             128      
   ConvBn-2     [[1, 32, 81, 324], [1]]   [[1, 32, 41, 324], [1]]          0       
  ConvStack-1   [[1, 1, 161, 970], [1]]   [[1, 32, 41, 324], [1]]          0       
   Linear-1        [[1, 324, 1312]]           [1, 324, 3072]           4,030,464   
 BatchNorm1D-1     [[1, 324, 3072]]           [1, 324, 3072]            12,288     
   Linear-2        [[1, 324, 1312]]           [1, 324, 3072]           4,030,464   
 BatchNorm1D-2     [[1, 324, 3072]]           [1, 324, 3072]            12,288     
   GRUCell-1    [[1, 3072], [1, 1024]]    [[1, 1024], [1, 1024]]      12,589,056   
     RNN-1                []            [[1, 324, 1024], [1, 1024]]        0       
     RNN-2                []            [[1, 324, 1024], [1, 1024]]        0       
 BiGRUWithBN-1   [[1, 324, 1312], [1]]     [[1, 324, 2048], [1]]           0       
   Linear-3        [[1, 324, 2048]]           [1, 324, 3072]           6,291,456   
 BatchNorm1D-3     [[1, 324, 3072]]           [1, 324, 3072]            12,288     
   Linear-4        [[1, 324, 2048]]           [1, 324, 3072]           6,291,456   
 BatchNorm1D-4     [[1, 324, 3072]]           [1, 324, 3072]            12,288     
   GRUCell-3    [[1, 3072], [1, 1024]]    [[1, 1024], [1, 1024]]      12,589,056   
     RNN-3                []            [[1, 324, 1024], [1, 1024]]        0       
     RNN-4                []            [[1, 324, 1024], [1, 1024]]        0       
 BiGRUWithBN-2   [[1, 324, 2048], [1]]     [[1, 324, 2048], [1]]           0       
   Linear-5        [[1, 324, 2048]]           [1, 324, 3072]           6,291,456   
 BatchNorm1D-5     [[1, 324, 3072]]           [1, 324, 3072]            12,288     
   Linear-6        [[1, 324, 2048]]           [1, 324, 3072]           6,291,456   
 BatchNorm1D-6     [[1, 324, 3072]]           [1, 324, 3072]            12,288     
   GRUCell-5    [[1, 3072], [1, 1024]]    [[1, 1024], [1, 1024]]      12,589,056   
     RNN-5                []            [[1, 324, 1024], [1, 1024]]        0       
     RNN-6                []            [[1, 324, 1024], [1, 1024]]        0       
 BiGRUWithBN-3   [[1, 324, 2048], [1]]     [[1, 324, 2048], [1]]           0       
  RNNStack-1     [[1, 324, 1312], [1]]     [[1, 324, 2048], [1]]           0       
   Linear-7        [[1, 324, 2048]]           [1, 324, 4324]           8,859,876   
=====================================================================================
Total params: 80,178,756
Trainable params: 80,104,772
Non-trainable params: 73,984
-------------------------------------------------------------------------------------
Input size (MB): 0.60
Forward/backward pass size (MB): 169.54
Params size (MB): 305.86
Estimated Total Size (MB): 475.99
-------------------------------------------------------------------------------------

Epoch 1: ExponentialDecay set learning rate to 0.00083.
[2021-06-02 16:56:46.309161] Train epoch 0, batch 0, loss: 88.549583
[2021-06-02 17:02:12.195596] Train epoch 0, batch 100, loss: 7.689158
[2021-06-02 17:07:31.956302] Train epoch 0, batch 200, loss: 7.131282
[2021-06-02 17:12:46.002733] Train epoch 0, batch 300, loss: 6.795293
[2021-06-02 17:18:01.956726] Train epoch 0, batch 400, loss: 6.985234
```

可以用使用`python train.py --help`命令查看各个参数的说明和默认值。
```shell
optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS           训练使用的GPU序号，使用英文逗号,隔开，如：0,1 默认: 0.
  --batch_size BATCH_SIZE
                        训练的批量大小 默认: 16.
  --num_workers NUM_WORKERS
                        读取数据的线程数量 默认: 8.
  --num_epoch NUM_EPOCH
                        训练的轮数 默认: 20.
  --learning_rate LEARNING_RATE
                        初始学习率的大小 默认: 0.001.
  --num_conv_layers NUM_CONV_LAYERS
                        卷积层数量 默认: 2.
  --num_rnn_layers NUM_RNN_LAYERS
                        循环神经网络的数量 默认: 3.
  --rnn_layer_size RNN_LAYER_SIZE
                        循环神经网络的大小 默认: 1024.
  --min_duration MIN_DURATION
                        过滤最短的音频长度 默认: 0.
  --max_duration MAX_DURATION
                        过滤最长的音频长度，当为-1的时候不限制长度 默认: 20.
  --train_manifest TRAIN_MANIFEST
                        训练数据的数据列表路径 默认: dataset/manifest.train.
  --test_manifest TEST_MANIFEST
                        测试数据的数据列表路径 默认: dataset/manifest.test.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/vocabulary.json.
  --mean_std_path MEAN_STD_PATH
                        数据集的均值和标准值的npy文件路径 默认: dataset/mean_std.npz.
  --save_model SAVE_MODEL
                        模型保存的路径 默认: models/.
  --resume RESUME       恢复训练，当为None则不使用预训练模型 默认: None.
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

在评估和预测中，使用`--decoder`参数可以指定解码方法，当`--decoder`参数为`ctc_greedy`对结果解码的贪心策略解码方法，贪心策略是在每一步选择概率最大的输出值，然后删除重复字符和空索引，就得到预测结果了。当`--decoder`参数为`ctc_beam_search`对结果解码的集束搜索解码方法，该方法可以加载语言模型，将模型输出的结果在语音模型中搜索最优解。

 - 我们可以使用这个脚本对模型进行评估，通过字符错误率来评价模型的性能。参数`--decoder`默认指定集束搜索解码方法对结果进行解码，读者也可以使用贪心策略解码方法，对比他们的解码的准确率。参数`--mean_std_path`指定均值和标准值得文件，这个文件需要跟训练时使用的是同一个文件。参数`--beam_size`指定集束搜索方法的搜索宽度，越大解码结果越准确，但是解码速度就越慢。参数`--model_path`指定模型所在的文件夹的路径。
```shell script
python3 eval.py --model_path=models/step_final/
```

可以用使用`python eval.py --help`命令查看各个参数的说明和默认值。
```shell
optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        训练的批量大小 默认: 32.
  --num_workers NUM_WORKERS
                        读取数据的线程数量 默认: 8.
  --num_conv_layers NUM_CONV_LAYERS
                        卷积层数量 默认: 2.
  --num_rnn_layers NUM_RNN_LAYERS
                        循环神经网络的数量 默认: 3.
  --rnn_layer_size RNN_LAYER_SIZE
                        循环神经网络的大小 默认: 1024.
  --alpha ALPHA         定向搜索的LM系数 默认: 1.2.
  --beta BETA           定向搜索的WC系数 默认: 0.35.
  --beam_size BEAM_SIZE
                        定向搜索的大小，范围:[5, 500] 默认: 10.
  --num_proc_bsearch NUM_PROC_BSEARCH
                        定向搜索方法使用CPU数量 默认: 8.
  --cutoff_prob CUTOFF_PROB
                        剪枝的概率 默认: 1.0.
  --cutoff_top_n CUTOFF_TOP_N
                        剪枝的最大值 默认: 40.
  --test_manifest TEST_MANIFEST
                        测试数据的数据列表路径 默认: dataset/manifest.test.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/vocabulary.json.
  --mean_std_path MEAN_STD_PATH
                        数据集的均值和标准值的npy文件路径 默认: dataset/mean_std.npz.
  --model_path MODEL_PATH
                        模型的路径 默认: models/step_final/.
  --decoder {ctc_beam_search,ctc_greedy}
                        结果解码方法 默认: ctc_beam_search.
  --lang_model_path LANG_MODEL_PATH
                        语言模型文件路径 默认: lm/zh_giga.no_cna_cmn.prune01244.klm.
```

 - 我们可以使用这个脚本使用模型进行预测，通过传递音频文件的路径进行识别。参数`--decoder`默认指定集束搜索解码方法对结果进行解码，读者也可以使用贪心策略解码方法，对比他们的解码的准确率。参数`--mean_std_path`指定均值和标准值得文件，这个文件需要跟训练时使用的是同一个文件。参数`--beam_size`指定集束搜索方法的搜索宽度，越大解码结果越准确，但是解码速度就越慢。参数`model_path`指定模型所在的文件夹的路径，参数`wav_path`指定需要预测音频文件的路径。
```shell script
python3 infer.py --audio_path=./dataset/test.wav
```

可以用使用`python infer.py --help`命令查看各个参数的说明和默认值。
```shell
optional arguments:
  -h, --help            show this help message and exit
  --num_conv_layers NUM_CONV_LAYERS
                        卷积层数量 默认: 2.
  --num_rnn_layers NUM_RNN_LAYERS
                        循环神经网络的数量 默认: 3.
  --rnn_layer_size RNN_LAYER_SIZE
                        循环神经网络的大小 默认: 1024.
  --alpha ALPHA         定向搜索的LM系数 默认: 1.2.
  --beta BETA           定向搜索的WC系数 默认: 0.35.
  --beam_size BEAM_SIZE
                        定向搜索的大小，范围:[5, 500] 默认: 10.
  --num_proc_bsearch NUM_PROC_BSEARCH
                        定向搜索方法使用CPU数量 默认: 8.
  --cutoff_prob CUTOFF_PROB
                        剪枝的概率 默认: 1.0.
  --cutoff_top_n CUTOFF_TOP_N
                        剪枝的最大值 默认: 40.
  --audio_path AUDIO_PATH
                        用于识别的音频路径 默认: dataset/test.wav.
  --dataset_vocab DATASET_VOCAB
                        数据字典的路径 默认: dataset/vocabulary.json.
  --model_path MODEL_PATH
                        模型的路径 默认: models/step_final/.
  --mean_std_path MEAN_STD_PATH
                        数据集的均值和标准值的npy文件路径 默认: dataset/mean_std.npz.
  --decoder {ctc_beam_search,ctc_greedy}
                        结果解码方法 默认: ctc_beam_search.
  --lang_model_path LANG_MODEL_PATH
                        语言模型文件路径 默认: lm/zh_giga.no_cna_cmn.prune01244.klm.
```

## 模型下载
| 数据集 | 字错率 | 下载地址 |
| :---: | :---: | :---: |
| AISHELL | 训练中 | [点击下载]() |
| free_st_chinese_mandarin_corpus | 训练中 | [点击下载]() |
| thchs30 | 训练中 | [点击下载]() |
