# 安装PPASR环境

本人用的就是本地环境和使用Anaconda，并创建了Python3.8的虚拟环境，出现安装问题，随时提[issue](https://github.com/yeyupiaoling/PPASR/issues)。

 - 首先安装的是PaddlePaddle 2.4.1的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

 - 安装PPASR库。
 
使用pip安装，命令如下：
```shell
python -m pip install ppasr -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。
```shell
git clone https://github.com/yeyupiaoling/PPASR.git
cd PPASR
python setup.py install
```

**常见安装问题：** 

1. LLVM版本错误，则执行下面的命令，然后重新执行上面的安装命令，否则不需要执行。
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

2. Linux 报错 OSError: sndfile library not found

```shell
sudo apt-get install libsndfile1
```

3. 如果提示缺少`itn`依赖库，请安装。
```shell
python -m pip install WeTextProcessing>=0.1.0
```

5. 安装pynini出错，可以执行下面命令安装。
```shell
conda install -c conda-forge pynini
```