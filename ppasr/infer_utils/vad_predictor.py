import os
from typing import List

import numpy as np
import onnxruntime
from ppasr.utils.logger import setup_logger

logger = setup_logger(__name__)


class VADPredictor(object):
    """
    语音活动检测工具，来源于：https://github.com/snakers4/silero-vad

    :param path: 模型路径
    """

    def __init__(self,
                 path=None,
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100,
                 window_size_samples: int = 512,
                 speech_pad_ms: int = 30):
        """

        :param path: 模型文件路径
        :param threshold: 语音活动检测的阈值
        :param min_speech_duration_ms: 检测的最短语音
        :param min_silence_duration_ms: 最小检测音频静音的长度
        :param window_size_samples: VAD模型训练使用采样率是16000，该参数是512、1024、1536，采样率是8000，该参数是256,512、768
        :param speech_pad_ms: 语音填充的长度
        """
        if path is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'silero_vad.onnx')
        self.session = onnxruntime.InferenceSession(path)
        self.session.intra_op_num_threads = 1
        self.session.inter_op_num_threads = 1

        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

        self.reset_states()
        self.sample_rates = [8000, 16000]
        logger.info('成功初始化VAD推理工具')

    def _validate_input(self, x, sr: int):
        if len(x.shape) == 1:
            x = x[np.newaxis, :]
        if len(x.shape) > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")

        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        self._h = np.zeros((2, batch_size, 64)).astype('float32')
        self._c = np.zeros((2, batch_size, 64)).astype('float32')
        self._last_sr = 0
        self._last_batch_size = 0

        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(self, x, sr: int):
        x, sr = self._validate_input(x, sr)
        batch_size = x.shape[0]

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and (self._last_sr != sr):
            self.reset_states(batch_size)
        if self._last_batch_size and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if sr in [8000, 16000]:
            ort_inputs = {'input': x, 'h': self._h, 'c': self._c, 'sr': np.array(sr, dtype=np.int64)}
            ort_outs = self.session.run(None, ort_inputs)
            out, self._h, self._c = ort_outs
        else:
            raise ValueError()

        self._last_sr = sr
        self._last_batch_size = batch_size

        return out

    def get_speech_timestamps(self, audio, sampling_rate):
        """

        :param audio: 一维的音频数据，数据类型为：np.float32
        :param sampling_rate: 音频采样率，只支持8K或者16K
        :return: 每一个活动的音语音片段的列表
        """
        self.reset_states()
        min_speech_samples = sampling_rate * self.min_speech_duration_ms / 1000
        min_silence_samples = sampling_rate * self.min_silence_duration_ms / 1000
        speech_pad_samples = sampling_rate * self.speech_pad_ms / 1000

        audio_length_samples = len(audio)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, self.window_size_samples):
            chunk = audio[current_start_sample: current_start_sample + self.window_size_samples]
            if len(chunk) < self.window_size_samples:
                chunk = np.pad(chunk, (0, int(self.window_size_samples - len(chunk))))
            speech_prob = self(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)

        triggered = False
        speeches: List[dict] = []
        current_speech = {}
        neg_threshold = self.threshold - 0.15
        temp_end = 0

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= self.threshold) and temp_end:
                temp_end = 0

            if (speech_prob >= self.threshold) and not triggered:
                triggered = True
                current_speech['start'] = self.window_size_samples * i
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = self.window_size_samples * i
                if (self.window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    temp_end = 0
                    current_speech = {}
                    triggered = False
                    continue

        if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
            current_speech['end'] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:
                    speech['end'] += int(silence_duration // 2)
                    speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - silence_duration // 2))
                else:
                    speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                    speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - speech_pad_samples))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

        return speeches

    def stream_vad(self, x, sampling_rate, return_seconds=False):
        """

        :param x: 一维的音频数据块
        :param sampling_rate: 音频采样率，只支持8K或者16K
        :param return_seconds: 是否以秒为单位返回时间戳
        :return:
        """
        if len(x) < self.window_size_samples:
            return None
        min_silence_samples = sampling_rate * self.min_silence_duration_ms / 1000
        speech_pad_samples = sampling_rate * self.speech_pad_ms / 1000
        window_size_samples = len(x[0]) if len(x.shape) == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self(x, sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - speech_pad_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + speech_pad_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / sampling_rate, 1)}

        return None
