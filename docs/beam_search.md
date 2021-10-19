# 集束搜索解码

本项目目前支持两种解码方法，分别是集束搜索(ctc_beam_search)和贪婪策略(ctc_greedy)，项目全部默认都是使用贪婪策略解码的，集束搜索解码只支持Linux，如果要使用集束搜索方法，首先要安装`ctc_decoders`库，项目中作者提供了编译好的`ctc_decoders`库，执行项目命令即可安装完成。
```shell
cd tools
pip3 install swig_decoders-1.2-cp37-cp37m-linux_x86_64.whl
```

如果不能正常安装，就需要自行编译`ctc_decoders`库，该编译只支持Ubuntu，其他Linux版本没测试过，执行下面命令完成编译。
```shell
cd decoders
sh setup.sh
```


# 语言模型

集束搜索解码需要使用到语言模型，下载语言模型并放在lm目录下，下面下载的小语言模型，如何有足够大性能的机器，可以下载70G的超大语言模型，点击下载[Mandarin LM Large](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm) ，这个模型会大超多。
```shell script
cd PPASR/
mkdir lm
cd lm
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
```

# 寻找最优的alpha和beta

这一步可以跳过，使用默认的alpha和beta也是不错的，如果想精益求精，可以执行下面的命令，可能速度会比较慢。执行完成之后会得到效果最好的alpha和beta参数值。
```shell
python tools/tune.py --resume_model=models/deepspeech2/epoch_50
```

# 使用集束搜索解码

在需要使用到解码器的程序，如评估，预测，指定参数`--decoding_method`为`ctc_beam_search`即可，如果alpha和beta参数值有改动，修改对应的值即可。
