import random

import numpy as np
from PIL import Image
from PIL.Image import BICUBIC


class SpecAugmentor(object):
    """Augmentation model for Time warping, Frequency masking, Time masking.

    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        https://arxiv.org/abs/1904.08779
    SpecAugment on Large Scale Datasets
        https://arxiv.org/abs/1912.05533
    """

    def __init__(self,
                 rng,
                 F=30,
                 T=40,
                 n_freq_masks=2,
                 n_time_masks=2,
                 inplace=True,
                 max_time_warp=5,
                 replace_with_zero=False):
        """SpecAugment class.
        Args:
            :param F: 频率屏蔽参数
            :type F: int
            :param T: 时间屏蔽参数
            :type T: int
            :param n_freq_masks: 频率屏蔽数量
            :type n_freq_masks: int
            :param n_time_masks: 时间屏蔽数量
            :type n_time_masks: int
            :param inplace: 用结果覆盖
            :type inplace: bool
            :param max_time_warp: 时间变形参数
            :type max_time_warp: int
            :param replace_with_zero: 如果真的话，在pad补0，否则使用平均值
            :type replace_with_zero: bool
        """
        super().__init__()
        self._rng = rng
        self.inplace = inplace
        self.replace_with_zero = replace_with_zero

        self.max_time_warp = max_time_warp
        self.F = F
        self.T = T
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __repr__(self):
        return f"specaug: F-{self.F}, T-{self.T}, F-n-{self.n_freq_masks}, T-n-{self.n_time_masks}"

    def time_warp(self, x, mode='PIL'):
        """time warp for spec augment
        move random center frame by the random width ~ uniform(-window, window)

        Args:
            x (np.ndarray): spectrogram (time, freq)
            mode (str): PIL or sparse_image_warp

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            np.ndarray: time warped spectrogram (time, freq)
        """
        window = self.max_time_warp
        if window == 0:
            return x

        t = x.shape[0]
        if t - window <= window:
            return x
        # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
        center = random.randrange(window, t - window)
        warped = random.randrange(center - window, center + window) + 1  # 1 ... t - 1
        left = Image.fromarray(x[:center]).resize((x.shape[1], warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped), BICUBIC)
        if self.inplace:
            x[:warped] = left
            x[warped:] = right
            return x
        return np.concatenate((left, right), 0)

    def freq_mask(self, x, replace_with_zero=False):
        """freq mask

        Args:
            x (np.ndarray): spectrogram (time, freq)
            replace_with_zero (bool, optional): Defaults to False.

        Returns:
            np.ndarray: freq mask spectrogram (time, freq)
        """
        if self.inplace:
            cloned = x
        else:
            cloned = x.copy()

        num_mel_channels = cloned.shape[1]
        fs = np.random.randint(0, self.F, size=(self.n_freq_masks, 2))

        for f, mask_end in fs:
            f_zero = random.randrange(0, num_mel_channels - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            if replace_with_zero:
                cloned[:, f_zero:mask_end] = 0
            else:
                cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned

    def mask_time(self, x, replace_with_zero=False):
        """time mask

        Args:
            x (np.ndarray): spectrogram (time, freq)
            replace_with_zero (bool, optional): Defaults to False.

        Returns:
            np.ndarray: time mask spectrogram (time, freq)
        """
        if self.inplace:
            cloned = x
        else:
            cloned = x.copy()
        len_spectro = cloned.shape[0]
        ts = np.random.randint(0, self.T, size=(self.n_time_masks, 2))
        for t, mask_end in ts:
            # avoid randint range error
            if len_spectro - t <= 0:
                continue
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            if replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned

    def __call__(self, x, train=True):
        if not train:
            return x
        return self.transform_feature(x)

    def transform_feature(self, x: np.ndarray):
        """
        Args:
            x (np.ndarray): `[T, F]`
        Returns:
            x (np.ndarray): `[T, F]`
        """
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        x = self.time_warp(x)
        x = self.freq_mask(x, self.replace_with_zero)
        x = self.mask_time(x, self.replace_with_zero)
        return x
