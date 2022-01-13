# Nvidia Jetson部署

1. 这对Nvidia Jetson设备，如Nano、Nx、AGX等设备，可以通过下面命令安装PaddlePaddle的Inference预测库。
```shell
# Jetpack4.4(4.5)版本
wget https://paddle-inference-lib.bj.bcebos.com/2.2.1/python/Jetson/jetpack4.4_gcc7.5/all/paddlepaddle_gpu-2.2.1-cp36-cp36m-linux_aarch64.whl
# Jetpack4.6版本
wget https://paddle-inference-lib.bj.bcebos.com/2.2.1/python/Jetson/jetpack4.6_gcc7.5/all/paddlepaddle_gpu-2.2.1-cp36-cp36m-linux_aarch64.whl
# 安装PaddlePaddle
pip3 install paddlepaddle_gpu-2.2.1-cp36-cp36m-linux_aarch64.whl
```

2. 安装scikit-learn依赖库。
```shell
git clone git://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
pip3 install cython
git checkout 0.24.2
pip3 install --verbose --no-build-isolation --editable .
```

3. 安装其他依赖库。
```shell
pip3 install -r requirements.txt
```

3. 执行预测，直接使用根目录下的预测代码。
```shell
python infer_path.py --wav_path=./dataset/test.wav
```

以Nvidia AGX为例，输出结果如下：
```
WARNING: AVX is not support on your machine. Hence, no_avx core will be imported, It has much worse preformance than avx core.
-----------  Configuration Arguments -----------
alpha: 2.2
beam_size: 300
beta: 4.3
cutoff_prob: 0.99
cutoff_top_n: 40
decoder: ctc_beam_search
is_long_audio: False
lang_model_path: lm/zh_giga.no_cna_cmn.prune01244.klm
model_dir: models/deepspeech2/infer/
pun_model_dir: models/pun_models/
real_time_demo: False
use_pun: False
to_an: True
use_gpu: True
use_model: deepspeech2
vocab_path: dataset/vocabulary.txt
wav_path: ./dataset/test.wav
------------------------------------------------
消耗时间：416ms, 识别结果: 近几年不但我用书给女儿压岁也劝说亲朋不要给女儿压岁钱而改送压岁书, 得分: 97
```