"""Contains the noise perturb augmentation model."""
import random

import numpy as np

from ppasr.data_utils.augmentor.base import AugmentorBase
from ppasr.data_utils.utils import read_manifest
from ppasr.data_utils.audio import AudioSegment


class NoisePerturbAugmentor(AugmentorBase):
    """用于添加背景噪声的增强模型

    :param min_snr_dB: Minimal signal noise ratio, in decibels.
    :type min_snr_dB: float
    :param max_snr_dB: Maximal signal noise ratio, in decibels.
    :type max_snr_dB: float
    :param repetition: repetition noise sum
    :type repetition: int
    :param noise_manifest_path: Manifest path for noise audio data.
    :type noise_manifest_path: str
    """

    def __init__(self, min_snr_dB, max_snr_dB, repetition, noise_manifest_path):
        self._min_snr_dB = min_snr_dB
        self._max_snr_dB = max_snr_dB
        self.repetition = repetition
        self._noise_manifest = read_manifest(manifest_path=noise_manifest_path)

    def transform_audio(self, audio_segment: AudioSegment):
        """Add background noise audio.

        Note that this is an in-place transformation.

        :param audio_segment: Audio segment to add effects to.
        :type audio_segment: AudioSegmenet
        """
        if len(self._noise_manifest) > 0:
            for _ in range(random.randint(1, self.repetition)):
                # 随机选择一个noises_path中的一个
                noise_json = random.sample(self._noise_manifest, 1)[0]
                # 读取噪声音频
                noise_segment = AudioSegment.from_file(noise_json['audio_filepath'])
                # 如果噪声采样率不等于audio_segment的采样率，则重采样
                if noise_segment.sample_rate != audio_segment.sample_rate:
                    noise_segment.resample(audio_segment.sample_rate)
                # 随机生成snr_dB的值
                snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
                # 如果噪声的长度小于audio_segment的长度，则将噪声的前面的部分填充噪声末尾补长
                if noise_segment.duration < audio_segment.duration:
                    diff_duration = audio_segment.num_samples - noise_segment.num_samples
                    noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
                # 将噪声添加到audio_segment中，并将snr_dB调整到最小值和最大值之间
                audio_segment.add_noise(noise_segment, snr_dB)
