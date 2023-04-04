# 集束搜索解码

本项目目前支持两种解码方法，分别是集束搜索(ctc_beam_search)和贪婪策略(ctc_greedy)，如果要使用集束搜索方法，首先要安装`paddlespeech-ctcdecoders`库，执行以下命令即可安装完成。
```shell
python -m pip install paddlespeech_ctcdecoders -U -i https://ppasr.yeyupiaoling.cn/pypi/simple/
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

在需要使用到解码器的程序，如评估，预测，在`configs/config_zh.yml`配置文件中修改参数`decoder`为`ctc_beam_search`即可，如果alpha和beta参数值有改动，修改对应的值即可。



# 语言模型表格

|                                          语言模型                                          |                                                      训练数据                                                       |  数据量  |  文件大小   |                 说明                  |
|:--------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:-----:|:-------:|:-----------------------------------:|
|         [自定义中文语言模型](https://pan.baidu.com/s/1vdQsqnoKHO9jdFU_1If49g?pwd=ea09)          |                       [自定义中文语料](https://download.csdn.net/download/qq_33200967/87002687)                        | 约2千万  | 572 MB  |           训练参数`-o 5`，无剪枝            |
|  [英文语言模型](https://deepspeech.bj.bcebos.com/en_lm/common_crawl_00.prune01111.trie.klm)  | [CommonCrawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz) | 18.5亿 | 8.3 GB  | 训练参数`-o 5`，剪枝参数`'--prune 0 1 1 1 1` |
| [中文语言模型（剪枝）](https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm) |                                                     百度内部语料库                                                     | 1.3亿  | 2.8 GB  | 训练参数`-o 5`，剪枝参数`'--prune 0 1 1 1 1` |                                     |
|            [中文语言模型](https://deepspeech.bj.bcebos.com/zh_lm/zhidao_giga.klm)            |                                                     百度内部语料库                                                     |  37亿  | 70.4 GB |           训练参数`-o 5`，无剪枝            |                                     


# 训练自己的语言模型

1. 首先安装kenlm，此步骤要从项目根目录开始，以下是Ubuntu的安装方式，其他系统请自行百度。
```shell
sudo apt install -y libbz2-dev liblzma-dev cmake build-essential libboost-all-dev
cd tools/
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
```

2. 准备kenlm语料，此步骤要从项目根目录开始，使用的语料是训练数据集，所以要执行`create_data.py`完成之后才能执行下面操作。或者自己准备语料，修改生成语料的代码。

```shell
cd tools/
python create_kenlm_corpus.py
```

3. 有了kenlm语料之后，就可以训练kenlm模型了，此步骤要从项目根目录开始，执行下面命令训练和压缩模型。
```shell
cd tools/kenlm/build/ 
bin/lmplz -o 5 --verbose header --text ../../../dataset/corpus.txt --arpa ../../../lm/my_lm.arpa
# 把模型转为二进制，减小模型大小
bin/build_binary trie -a 22 -q 8 -b 8 ../../../lm/my_lm.arpa ../../../lm/my_lm.klm
```

4. 可以使用下面代码测试模型是否有效。
```python
import kenlm

model = kenlm.Model('kenlm1/build/model/test.klm')
result = model.score('近几年不但我用书给女儿儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书', bos=True, eos=True)
print(result)
```