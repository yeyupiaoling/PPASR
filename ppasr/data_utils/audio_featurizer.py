import numpy as np
import paddle
from paddleaudio.compliance.kaldi import mfcc, fbank, spectrogram


class AudioFeaturizer(object):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: Any
    :param mode: 使用模式
    :type mode: str
    """

    def __init__(self, feature_method='fbank', method_args=None, mode="train"):
        assert feature_method in ['fbank', 'mfcc', 'spectrogram'], f'没有{feature_method}预处理方法'
        self._feature_method = feature_method
        self._mode = mode
        self._method_args = method_args

    def featurize(self, waveform, sample_rate):
        """计算音频特征

        :param waveform: 音频数据
        :type waveform: paddle.Tensor
        :param sample_rate: 音频采样率
        :type sample_rate: int
        :return: 二维的音频特征
        :rtype: paddle.Tensor
        """
        if isinstance(waveform, np.ndarray):
            waveform = paddle.to_tensor(waveform, dtype=paddle.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if self._mode == 'train':
            self._method_args.dither = 0.0
        # 计算音频特征
        if self._feature_method == 'spectrogram':
            # 计算Spectrogram
            feature = spectrogram(waveform, sr=sample_rate, **self._method_args)
        elif self._feature_method == 'mfcc':
            # 计算MFCC
            feature = mfcc(waveform, sr=sample_rate, **self._method_args)
        elif self._feature_method == 'fbank':
            # 计算Fbank
            feature = fbank(waveform, sr=sample_rate, **self._method_args)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'spectrogram':
            feature = spectrogram(paddle.ones(16000), **self._method_args)
            return feature.shape[1]
        elif self._feature_method == 'mfcc':
            return self._method_args.get('n_mfcc', 13)
        elif self._feature_method == 'fbank':
            return self._method_args.get('n_mels', 23)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
