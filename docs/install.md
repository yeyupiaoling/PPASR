# 搭建本地环境

本人用的就是本地环境和使用Anaconda，并创建了Python3.7的虚拟环境，建议读者也本地环境，方便交流，出现安装问题，随时提[issue](https://github.com/yeyupiaoling/PPASR/issues) ，如果想使用docker，请查看**搭建Docker环境**。

 - 首先安装的是PaddlePaddle 2.1.3的GPU版本，如果已经安装过了，请跳过。
```shell
conda install paddlepaddle-gpu==2.1.3 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

 - 安装其他依赖库。
```shell
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**注意：** 如果出现LLVM版本错误，解决办法[LLVM版本错误](./faq.md)。


# 搭建Docker环境

 - 请提前安装好显卡驱动，然后执行下面的命令。
```shell script
# 卸载系统原有docker
sudo apt-get remove docker docker-engine docker.io containerd runc
# 更新apt-get源 
sudo apt-get update
# 安装docker的依赖 
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
# 添加Docker的官方GPG密钥：
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# 验证拥有指纹
sudo apt-key fingerprint 0EBFCD88
# 设置稳定存储库
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```

 - 安装Docker
```shell script
# 再次更新apt-get源 
sudo apt-get update
# 开始安装docker 
sudo apt-get install docker-ce
# 加载docker 
sudo apt-cache madison docker-ce
# 验证docker是否安装成功
sudo docker run hello-world
```

 - 安装nvidia-docker
```shell script
# 设置stable存储库和GPG密钥
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 更新软件包清单后
sudo apt-get update

# 安装软件包
sudo apt-get install -y nvidia-docker2

# 设置默认运行时后，重新启动Docker守护程序以完成安装：
sudo systemctl restart docker

# 测试
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

 - 拉取PaddlePaddle 2.1.2镜像。
```shell script
sudo nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7
```

- git clone 本项目源码
```shell script
git clone https://github.com/yeyupiaoling/DeepSpeech.git
```

- 运行PaddlePaddle语音识别镜像，这里设置与主机共同拥有IP和端口号。
```shell script
sudo nvidia-docker run -it --net=host -v $(pwd)/DeepSpeech:/DeepSpeech registry.baidubce.com/paddlepaddle/paddle:2.1.2-gpu-cuda10.2-cudnn7 /bin/bash
```

 - 安装其他依赖库。
```shell
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
