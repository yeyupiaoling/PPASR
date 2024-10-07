# 怎么设计词汇表

1. 如果是纯中文数据，可以把`model_type`设置为`char`，然后`build_vocab_size`设置为`null`，这样会使用数据集出现的全部字符。
2. 如果是英文数据，也可以把`model_type`设置为`word`，然后`build_vocab_size`设置为`null`，这样会使用数据集出现的全部单词，但是不建议这么做，因为这样的词汇表会变成的非常大，不利于训练，例如Librispeech数据集就会有9万多个单词，所以建议使用把`model_type`设置为`unigram`，然后`build_vocab_size`设置为`5000`，也可以跟更大一些，根据数据集量设置。
3. 如果是中混合数据，可以把`model_type`设置为`unigram`，然后`build_vocab_size`设置为`10000`左右。
4. 如果是其他语言，可以直接使用`model_type`=`unigram`，然后`build_vocab_size`设置为`10000`左右，如果报错太大了，可以更加提示设置小一些。


# 要训练多少个epoch

1. 这个不是固定的，项目默认训练是200轮，但是也可以根据自己的数据训练的收敛情况而提前终止。可以通过VisualDL的趋势图分析，如下图，输出的Loss和错误率已经很平滑了，就可以提前结束训练。当然如果在不缺算力的情况下最好训练完整，因为在训练过程中由数据增强，一定程度上可以提高模型的泛化能力。

<div align="center">
  <img src="./images/visualdl.jpg" alt="VisualDL" width="800">
</div>

