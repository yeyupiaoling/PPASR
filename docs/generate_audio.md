# 合成语音数据

1. 为了拟补数据集的不足，我们合成一批语音用于训练，语音合成一批音频文件。首先安装PaddleSpeech，执行下面命令即可安装完成。
```shell
python -m pip install paddlespeech
```

2. 然后下载一个语料，如果开发者有其他更好的语料也可以替换。然后解压`dgk_lost_conv/results`目录下的压缩文件，windows用户可以手动解压。
```shell
cd tools/generate_audio
git clone https://github.com/aceimnorstuvwxz/dgk_lost_conv.git
cd dgk_lost_conv/results
unzip dgk_shooter_z.conv.zip
unzip xiaohuangji50w_fenciA.conv.zip
unzip xiaohuangji50w_nofenci.conv.zip
```

3. 接着执行下面命令生成中文语料数据集，生成的中文语料存放在`tools/generate_audio/corpus.txt`。
```shell
cd tools/generate_audio/
python generate_corpus.py
```

4. 最后执行以下命令即可自动合成语音，合成时会随机获取说话人进行合成语音，合成的语音会放在`dataset/audio/generate`， 标注文件会放在`dataset/annotation/generate.txt`。
```shell
cd tools/generate_audio/
python generate_audio.py
```
