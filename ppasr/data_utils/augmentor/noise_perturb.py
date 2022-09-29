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
        :type audio_segment: AudioSegmenet|SpeechSegment
        """
        for _ in range(random.randint(1, self.repetition)):
            noise_json = random.sample(self._noise_manifest, 1)[0]
            noise_segment = AudioSegment.from_file(noise_json['audio_filepath'])
            snr_dB = random.uniform(self._min_snr_dB, self._max_snr_dB)
            if noise_segment.samples.shape[0] < audio_segment.samples.shape[0]:
                diff_duration = audio_segment.samples.shape[0] - noise_segment.samples.shape[0]
                noise_segment._samples = np.pad(noise_segment.samples, (0, diff_duration), 'wrap')
            audio_segment.add_noise(noise_segment, snr_dB, allow_downsampling=True, rng=self._rng)
