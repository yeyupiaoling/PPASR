import numpy as np
import resampy
import soundfile
from python_speech_features import mfcc
from python_speech_features import delta
from .audio_tool import AudioTool


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
    :param target_db: 目标音频分贝为标准化
    :type target_db: float
    """

    def __init__(self,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 target_audio_rate=16000,
                 target_db=-20):
        self._stride_ms = stride_ms
        self._window_ms = window_ms
        self._max_freq = max_freq
        self._target_audio_rate = target_audio_rate
        self._target_dB = target_db
        self._audio_tool = AudioTool()

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
        audio = self._audio_tool.normalize(audio=audio, target_db=self._target_dB)
        # 计算音频梅尔频谱倒谱系数（MFCCs）
        audio = self._compute_mfcc(audio, self._target_audio_rate, self._stride_ms, self._window_ms, self._max_freq)
        return audio

    # 计算音频梅尔频谱倒谱系数（MFCCs）
    def _compute_mfcc(self,
                      samples,
                      sample_rate,
                      stride_ms=10.0,
                      window_ms=20.0,
                      max_freq=None):
        """Compute mfcc from samples."""
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than window size.")
        # 计算13个倒谱系数，第一个用log(帧能量)代替
        mfcc_feat = mfcc(signal=samples,
                         samplerate=sample_rate,
                         winlen=0.001 * window_ms,
                         winstep=0.001 * stride_ms,
                         highfreq=max_freq)
        # Deltas
        d_mfcc_feat = delta(mfcc_feat, 2)
        # Deltas-Deltas
        dd_mfcc_feat = delta(d_mfcc_feat, 2)
        # 转置
        mfcc_feat = np.transpose(mfcc_feat)
        d_mfcc_feat = np.transpose(d_mfcc_feat)
        dd_mfcc_feat = np.transpose(dd_mfcc_feat)
        # 拼接以上三个特点
        concat_mfcc_feat = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat))
        return concat_mfcc_feat

    @property
    def feature_dim(self):
        """返回特征的维度大小"""
        return 39
