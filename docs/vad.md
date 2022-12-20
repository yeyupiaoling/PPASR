# 语音活动检测（VAD）

针对长语音识别，本项目提供了一个语音活动检测的预测程序，通过这个预测器可以对长语音进行分割。通过检测静音的位置，把长语音分割成多段短语音。然后把分割后的音频通过短语音识别的方式来实现来进行识别。

这个语音活动检测预测器是使用onnxruntime推理的，所以在使用的前提要按照这个库。
```shell
python -m pip install onnxruntime
```

在本项目的使用可以参考`infer_path.py`的长语音识别，相关文档在[本地预测](./infer.md)。

如果想要单独使用语音活动检测的话，可以参考一下代码，注意输入的数据`wav`是`np.float32`的，因为输入的音频采样率只能是8K或者16K。
```python
import numpy as np
import soundfile

from ppasr.infer_utils.vad_predictor import VADPredictor

vad_predictor = VADPredictor()

wav, sr = soundfile.read('dataset/test_long.wav', dtype=np.float32)
speech_timestamps = vad_predictor.get_speech_timestamps(wav, sr)
for t in speech_timestamps:
    crop_wav = wav[t['start']: t['end']]
    print(crop_wav.shape)
```

# 流式实时语音活动检测（VAD）
最新版本可以支持流式检测语音活动，在录音的时候可以试试检测是否停止说话，从而完成一些业务，如停止录音开始识别等。
```python
import numpy as np
import soundfile

from ppasr.infer_utils.vad_predictor import VADPredictor

vad = VADPredictor()

wav, sr = soundfile.read('dataset/test.wav', dtype=np.float32)

for i in range(0, len(wav), vad.window_size_samples):
    chunk_wav = wav[i: i + vad.window_size_samples]
    speech_dict = vad.stream_vad(chunk_wav, sampling_rate=sr)
    if speech_dict:
        print(speech_dict, end=' ')
```

实时输出检测结果：
```
{'start': 11296} {'end': 21984} {'start': 25632} {'end': 54752} {'start': 57376} {'end': 97760} {'start': 103456} {'end': 124896} 
```
