# WenetSpeech数据集

[WenetSpeech数据集](https://wenet-e2e.github.io/WenetSpeech/)包含了10000+小时的普通话语音数据集，本教程介绍如何使用该数据集训练语音识别模型，主要分三步。

1. 下载并解压WenetSpeech数据集，在[官网](https://wenet-e2e.github.io/WenetSpeech/#download)填写表单之后，会收到邮件，执行邮件上面的三个命令就可以下载并解压数据集了，注意这要500G的磁盘空间。

2. 然后制作数据集，下载原始的数据是没有裁剪的，我们需要根据JSON标注文件裁剪并标注音频文件。在`tools`目录下执行`create_wenetspeech_data.py`程序就可以制作数据集了，注意此时需要3T的磁盘空间。`--wenetspeech_json`参数是指定WenetSpeech数据集的标注文件路径，具体根据读者下载的地址设置。
```shell
cd tools/
python create_wenetspeech_data.py --wenetspeech_json=/media/wenetspeech/WenetSpeech.json
```

3. 最后创建训练数据，跟普通使用一样，在项目根目录执行`create_data.py`就能过生成训练所需的数据列表，词汇表和均值标准差文件。这一步结束后就可以训练模型了，具体看[训练模型](./train.md)
```shell
python create_data.py
```

**温馨提示：** 数据集超大，费时费资源，看自己的情况使用，无金刚钻就不要揽瓷器活。