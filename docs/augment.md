# 数据增强

数据增强是用来提升深度学习性能的非常有效的技术。通过在原始音频中添加小的随机扰动（标签不变转换）获得新音频来增强的语音数据。开发者不必自己合成，因为数据增强已经嵌入到数据生成器中并且能够即时完成，在训练模型的每个epoch中随机合成音频。

目前提供五个可选的增强组件供选择，配置并插入处理过程。

- 噪声干扰（需要背景噪音的音频文件）
- 随机采样率增强
- 速度扰动
- 移动扰动
- 音量扰动
- SpecAugment增强方式
- SpecSubAugment增强方式

为了让训练模块知道需要哪些增强组件以及它们的处理顺序，需要事先准备一个JSON格式的*扩展配置文件*。例如：

```json
[
  {
    "type": "noise",
    "aug_type": "audio",
    "params": {
      "min_snr_dB": 10,
      "max_snr_dB": 50,
      "repetition": 2,
      "noise_manifest_path": "dataset/manifest.noise"
    },
    "prob": 0.5
  },
  {
    "type": "resample",
    "aug_type": "audio",
    "params": {
      "new_sample_rate": [8000, 32000, 44100, 48000]
    },
    "prob": 0.0
  },
  {
    "type": "speed",
    "aug_type": "audio",
    "params": {
      "min_speed_rate": 0.9,
      "max_speed_rate": 1.1,
      "num_rates": 3
    },
    "prob": 1.0
  },
  {
    "type": "shift",
    "aug_type": "audio",
    "params": {
      "min_shift_ms": -5,
      "max_shift_ms": 5
    },
    "prob": 1.0
  },
  {
    "type": "volume",
    "aug_type": "audio",
    "params": {
      "min_gain_dBFS": -15,
      "max_gain_dBFS": 15
    },
    "prob": 1.0
  },
  {
    "type": "specaug",
    "aug_type": "feature",
    "params": {
      "inplace": true,
      "max_time_warp": 5,
      "max_t_ratio": 0.05,
      "n_freq_masks": 2,
      "max_f_ratio": 0.15,
      "n_time_masks": 2,
      "replace_with_zero": false
    },
    "prob": 1.0
  },
  {
    "type": "specsub",
    "aug_type": "feature",
    "params": {
      "max_t": 30,
      "num_t_sub": 3
    },
    "prob": 1.0
  }
]
```

当`train.py`的`--augment_conf_file`参数被设置为上述示例配置文件的路径时，每个epoch中的每个音频片段都将被处理。首先，均匀随机采样速率会有50％的概率在 0.95 和 1.05
之间对音频片段进行速度扰动。然后，音频片段有 50％ 的概率在时间上被挪移，挪移偏差值是 -5 毫秒和 5 毫秒之间的随机采样。最后，这个新合成的音频片段将被传送给特征提取器，以用于接下来的训练。

使用数据增强技术时要小心，由于扩大了训练和测试集的差异，不恰当的增强会对训练模型不利，导致训练和预测的差距增大。

