# 集束搜索解码

本项目目前支持两种解码方法，分别是集束搜索(ctc_beam_search)和贪婪策略(ctc_greedy)，项目全部默认都是使用贪婪策略解码的，集束搜索解码只支持Linux且Python为3.7、3.8、3.9的，如果要使用集束搜索方法，首先要安装`swig_decoders`库，执行以下命令即可安装完成。
```shell
python -m pip install paddlespeech-ctcdecoders==0.1.1 -i https://mirrors.aliyun.com/pypi/simple/
```

# 语言模型

集束搜索解码需要使用到语言模型，在执行程序的时候，回自动下载语言模型，不过下载的是小语言模型，如何有足够大性能的机器，可以手动下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，并指定语言模型的路径。

注意，上面提到的语言模型都是中文语言模型，如果需要使用英文语言模型，需要手动下载，并指定语言模型路径。
```shell
https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm
```

# 寻找最优的alpha和beta

这一步可以跳过，使用默认的alpha和beta也是不错的，如果想精益求精，可以执行下面的命令，可能速度会比较慢。执行完成之后会得到效果最好的alpha和beta参数值。
```shell
python tools/tune.py --resume_model=models/deepspeech2/epoch_50
```

# 使用集束搜索解码

在需要使用到解码器的程序，如评估，预测，指定参数`--decoder`为`ctc_beam_search`即可，如果alpha和beta参数值有改动，修改对应的值即可。
