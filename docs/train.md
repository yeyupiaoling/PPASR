# 训练模型

 - 训练流程，首先是准备数据集，具体看[数据准备](./dataset.md)部分，重点是执行`create_data.py`程序，执行完成之后检查是否在`dataset`目录下生成了`manifest.test`、`manifest.train`、`mean_std.npz`、`vocabulary.txt`这四个文件，并确定里面已经包含数据。然后才能往下执行开始训练。

 - 执行训练脚本，开始训练语音识别模型，详细参数请查看该程序。每训练一轮和每10000个batch都会保存一次模型，模型保存在`PPASR/models/<use_model>/epoch_*/`目录下，默认会使用数据增强训练，如何不想使用数据增强，只需要将参数`augment_conf_path`设置为`None`即可。关于数据增强，请查看[数据增强](./faq.md)部分。如果没有关闭测试，在每一轮训练结果之后，都会执行一次测试计算模型在测试集的准确率。
```shell
# 单卡训练
python3 train.py
# 多卡训练
python -m paddle.distributed.launch --gpus '0,1' train.py
```

训练输出结果如下：
```
-----------  Configuration Arguments -----------
augment_conf_path: conf/augmentation.json
batch_size: 32
dataset_vocab: dataset/vocabulary.txt
learning_rate: 0.001
max_duration: 20
mean_std_path: dataset/mean_std.npz
min_duration: 0
num_epoch: 50
num_workers: 8
pretrained_model: None
resume_model: None
save_model_path: models/
test_manifest: dataset/manifest.test
train_manifest: dataset/manifest.train
use_model: deepspeech2
------------------------------------------------
............
[2021-09-17 08:41:16.135825] Train epoch: [24/50], batch: [5900/6349], loss: 3.84609, learning rate: 0.00000688, eta: 10:38:40
[2021-09-17 08:41:38.698795] Train epoch: [24/50], batch: [6000/6349], loss: 0.92967, learning rate: 0.00000688, eta: 8:42:11
[2021-09-17 08:42:04.166192] Train epoch: [24/50], batch: [6100/6349], loss: 2.05670, learning rate: 0.00000688, eta: 10:59:51
[2021-09-17 08:42:26.471328] Train epoch: [24/50], batch: [6200/6349], loss: 3.03502, learning rate: 0.00000688, eta: 11:51:28
[2021-09-17 08:42:50.002897] Train epoch: [24/50], batch: [6300/6349], loss: 2.49653, learning rate: 0.00000688, eta: 12:01:30

 ======================================================================
[2021-09-17 08:43:01.954403] Test batch: [0/65], loss: 13.76276, cer: 0.23105
[2021-09-17 08:43:07.817434] Test epoch: 24, time/epoch: 0:24:30.756875, loss: 6.90274, cer: 0.15213
====================================================================== 
```


 - 在训练过程中，程序会使用VisualDL记录训练结果，可以通过以下的命令启动VisualDL。
```shell
visualdl --logdir=log --host=0.0.0.0
```

 - 然后再浏览器上访问`http://localhost:8040`可以查看结果显示，如下。

![Learning rate](https://img-blog.csdnimg.cn/20210318165719805.png)
![Test Cer](https://s3.ax1x.com/2021/03/01/6PJaZV.jpg)
![Train Loss](https://s3.ax1x.com/2021/03/01/6PJNq0.jpg)