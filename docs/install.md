# 安装PPASR环境

本人用的就是本地环境和使用Anaconda，并创建了Python3.8的虚拟环境，出现安装问题，随时提[issue](https://github.com/yeyupiaoling/PPASR/issues)。

 - 首先安装的是PaddlePaddle 2.3.2的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=11.6 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge 
```

**注意：**Windows要安装这个版本，以上版本在Windows是有问题的。
```shell
conda install paddlepaddle-gpu==2.3.2 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

 - 安装PPASR库。
 
使用pip安装，命令如下：
```shell
python -m pip install ppasr -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

源码安装，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/PPASR.git
cd PPASR
python setup.py install
```

**注意：** 如果出现LLVM版本错误，解决办法[LLVM版本错误](./faq.md)。
