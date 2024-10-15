import kaldi_native_fbank as knf
import numpy as np
import torch


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
        self._method_opts = self.create_method_fn()

    def create_method_fn(self):
        if self._feature_method == 'mfcc':
            opts = knf.MfccOptions()
            opts.energy_floor = 1.0
            if self._mode != "train":
                opts.frame_opts.dither = 0
            # 默认参数
            opts.frame_opts.samp_freq = self._method_args.get('samp_freq', 16000)
            opts.num_ceps = self._method_args.get('num_ceps', 13)
        elif self._feature_method == 'fbank':
            opts = knf.FbankOptions()
            opts.energy_floor = 1.0
            if self._mode != "train":
                opts.frame_opts.dither = 0
            # 默认参数
            opts.frame_opts.samp_freq = self._method_args.get('samp_freq', 16000)
            opts.mel_opts.num_bins = self._method_args.get('num_mel_bins', 80)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
        # 把self._method_args中的参数赋值到opts中
        for key, value in self._method_args.items():
            if isinstance(value, dict) and hasattr(opts, key):
                for subkey, subvalue in value.items():
                    if hasattr(getattr(opts, key), subkey):
                        setattr(getattr(opts, key), subkey, subvalue)
            else:
                if hasattr(opts, key):
                    setattr(opts, key, value)
        return opts

    def featurize(self, waveform, sample_rate):
        """计算音频特征

        :param waveform: 音频数据
        :type waveform: torch.Tensor
        :param sample_rate: 音频采样率
        :type sample_rate: int
        :return: 二维的音频特征
        :rtype: torch.Tensor
        """
        if waveform.ndim != 1:
            assert waveform.ndim == 1, f'输入的音频数据必须是一维的，但是现在是{waveform.ndim}维'
        if self._feature_method == 'mfcc':
            method_fn = knf.OnlineMfcc(self._method_opts)
        elif self._feature_method == 'fbank':
            method_fn = knf.OnlineFbank(self._method_opts)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
        # 计算音频特征
        method_fn.accept_waveform(sample_rate, waveform.tolist())
        frames = method_fn.num_frames_ready
        feature = np.empty([frames, self.feature_dim], dtype=np.float32)
        for i in range(method_fn.num_frames_ready):
            feature[i, :] = method_fn.get_frame(i)
        return feature

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'mfcc':
            return self._method_args.get('num_ceps', 13)
        elif self._feature_method == 'fbank':
            return self._method_args.get('num_mel_bins', 80)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))
