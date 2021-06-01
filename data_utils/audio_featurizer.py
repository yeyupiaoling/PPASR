import numpy as np
import resampy
import soundfile
from python_speech_features import mfcc
from python_speech_features import delta


class AudioFeaturizer(object):
    """音频特征器,用于从AudioSegment或SpeechSegment内容中提取特性。

    Currently, it supports feature types of linear spectrogram and mfcc.

    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: When specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned; when specgram_type is 'mfcc', max_feq is the
                     highest band edge of mel filters.
    :types max_freq: None|float
    :param target_audio_rate: Audio are resampled (if upsampling or
                               downsampling is allowed) to this before
                               extracting spectrogram features.
    :type target_audio_rate: float
    :param use_dB_normalization: Whether to normalize the audio to a certain
                                 decibels before extracting the features.
    :type use_dB_normalization: bool
    :param target_dB: Target audio decibels for normalization.
    :type target_dB: float
    """

    def __init__(self,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 target_audio_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20):
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_audio_rate = target_audio_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB

    def load_audio_file(self, path):
        audio, audio_rate = soundfile.read(path, dtype='float32')
        if audio_rate != self._target_audio_rate:
            audio = resampy.resample(audio, audio_rate, self._target_audio_rate, filter=filter)
        return audio

    def featurize(self, audio):
        """从AudioSegment或SpeechSegment中提取音频特征

        :param audio: soundfile to extract features from.
        :type audio: numpy

        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        :raises ValueError: If audio sample rate is not supported.
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

    def _compute_mfcc(self,
                      audio,
                      audio_rate,
                      stride_ms=10.0,
                      window_ms=20.0,
                      max_freq=None):
        """Compute mfcc from samples."""
        if max_freq is None:
            max_freq = audio_rate / 2
        if max_freq > audio_rate / 2:
            raise ValueError("max_freq must not be greater than half of sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than window size.")
        # compute the 13 cepstral coefficients, and the first one is replaced
        # by log(frame energy)
        mfcc_feat = mfcc(signal=audio,
                         samplerate=audio_rate,
                         winlen=0.001 * window_ms,
                         winstep=0.001 * stride_ms,
                         highfreq=max_freq)
        # Deltas
        d_mfcc_feat = delta(mfcc_feat, 2)
        # Deltas-Deltas
        dd_mfcc_feat = delta(d_mfcc_feat, 2)
        # transpose
        mfcc_feat = np.transpose(mfcc_feat)
        d_mfcc_feat = np.transpose(d_mfcc_feat)
        dd_mfcc_feat = np.transpose(dd_mfcc_feat)
        # concat above three features
        concat_mfcc_feat = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat))
        return concat_mfcc_feat
