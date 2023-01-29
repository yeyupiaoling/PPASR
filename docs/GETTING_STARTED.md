# 快速使用

### 准备数据

1. 在`download_data`目录下载一个小的数据集。
```shell script
cd download_data/
python thchs_30.py
```

**注意：** 以上代码只支持在Linux下执行，**如果是Windows的话**，可以获取程序中的`DATA_URL`的下载地址单独下载，建议用迅雷等下载工具，这样下载速度快很多。然后把`download()`函数改为文件的绝对路径，如下。
```python
# 把这行代码
filepath = download(url, md5sum, target_dir)
# 修改为
filepath = "D:\\Download\\data_thchs30.tgz"
```

2. 最后执行下面的数据集处理程序。
```python
python create_data.py
```

### 训练模型

执行下面命令开始训练。
```shell
python train.py
```


### 评估

执行下面这个脚本对模型进行评估，通过字符错误率来评价模型的性能。
```shell
python eval.py --resume_model=models/conformer_streaming_fbank/best_model/
```

### 导出模型

导出为预测模型。
```shell
python export_model.py --resume_model=models/conformer_streaming_fbank/best_model/
```

### 预测

预测音频文件。
```shell script
python infer_path.py --wav_path=./dataset/test.wav
```
