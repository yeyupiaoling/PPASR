import numpy as np
import resampy
import soundfile


class AudioFeaturizer(object):
    """音频特征器

    :param stride_ms: 用于生成帧的步长大小(单位毫秒)
    :type stride_ms: float
    :param window_ms: 生成帧的窗口大小(单位毫秒)
    :type window_ms: float
    :param max_freq: 只返回采样率在[0,max_freq]之间的FFT
    :types max_freq: None|float
    :param target_audio_rate: 指定训练音频的采样率
    :type target_audio_rate: float
    """

    def __init__(self,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 target_audio_rate=16000):
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_audio_rate = target_audio_rate

    def load_audio_file(self, path):
        audio, audio_rate = soundfile.read(path, dtype='float32')
        if audio_rate != self._target_audio_rate:
            audio = resampy.resample(audio, audio_rate, self._target_audio_rate, filter=filter)
        return audio

    def featurize(self, audio):
        """audio中提取音频特征

        :param audio: 使用soundfile读取得到的数据
        :type audio: numpy

        :return: 经过处理的二维特征
        :rtype: ndarray
        """
        # extract spectrogram
        return self._compute_linear_specgram(audio, self._target_audio_rate, self._stride_ms, self._window_ms, self._max_freq)

    def _compute_linear_specgram(self,
                                 samples,
                                 sample_rate,
                                 stride_ms=10.0,
                                 window_ms=20.0,
                                 max_freq=None,
                                 eps=1e-14):
        """用 FFT energy计算线性谱图"""
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq不能大于采样率的一半")
        if stride_ms > window_ms:
            raise ValueError("stride_ms不能大于window_ms")
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)
        specgram, freqs = self._specgram_real(samples,
                                              window_size=window_size,
                                              stride_size=stride_size,
                                              sample_rate=sample_rate)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        return np.log(specgram[:ind, :] + eps)

    def _specgram_real(self, samples, window_size, stride_size, sample_rate):
        """计算来自真实信号的频谱图样本"""
        # extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
        assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])
        # window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]
        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)
        fft = fft ** 2
        scale = np.sum(weighting ** 2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale
        # prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        return fft, freqs

    @staticmethod
    def feature_dim():
        """返回特征的维度大小"""
        return 161
