# 前言

在语音识别中，模型输出的结果只是单纯的文本结果，并没有根据语法添加标点符号，本教程就是针对这种情况，在语音识别文本中根据语法情况加入标点符号，使得语音识别系统能够输出在标点符号的最终结果。

# 使用

使用主要分为三4步：

1. 首先是[下载七个标点的模型](https://download.csdn.net/download/qq_33200967/75664996)或者[下载三个标点符号的模型](https://download.csdn.net/download/qq_33200967/86539773)，个人偏向第二个模型，并解压到`models/`目录下，注意这个模型只支持中文，如果想自己训练模型的话，可以在[PunctuationModel](https://github.com/yeyupiaoling/PunctuationModel)训练模型，然后导出模型复制到`models/`目录。


2. 需要使用PaddleNLP工具，所以需要提前安装PaddleNLP，安装命令如下：

```shell
python -m pip install paddlenlp -i https://mirrors.aliyun.com/pypi/simple/ -U
```

3. 在使用时，将`use_pun`参数设置为True，输出的结果就自动加上了标点符号，如下。

```
消耗时间：101, 识别结果: 近几年，不但我用输给女儿压岁，也劝说亲朋，不要给女儿压岁钱，而改送压岁书。, 得分: 94
```

# 单独使用标点符号模型

如果只是使用标点符号模型的话，可以参考一下代码。
```python
from ppasr.infer_utils.pun_predictor import PunctuationPredictor

pun_predictor = PunctuationPredictor(model_dir='models/pun_models')
result = pun_predictor('近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书')
print(result)
```

输出结果：
```
[2022-01-13 15:27:11,194] [    INFO] - Found C:\Users\test\.paddlenlp\models\ernie-1.0\vocab.txt
近几年，不但我用书给女儿儿压岁，也劝说亲朋，不要给女儿压岁钱，而改送压岁书。
```