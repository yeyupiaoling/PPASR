"""Contains the speech featurizer class."""

from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer


class SpeechFeaturizer(object):
    """Speech featurizer, for extracting features from both audio and transcript
    contents of SpeechSegment.

    Currently, for audio parts, it supports feature types of linear
    spectrogram and mfcc; for transcript parts, it only supports char-level
    tokenizing and conversion into a list of token indices. Note that the
    token indexing order follows the given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices
                           conversion.
    :type vocab_filepath: str
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param target_sample_rate: Speech are resampled (if upsampling or
                               downsampling is allowed) to this before
                               extracting spectrogram features.
    :type target_sample_rate: int
    :param use_dB_normalization: Whether to normalize the audio to a certain
                                 decibels before extracting the features.
    :type use_dB_normalization: bool
    :param target_dB: Target audio decibels for normalization.
    :type target_dB: float
    """

    def __init__(self, vocab_filepath, feature_method='linear', target_sample_rate=16000, target_dB=-20):
        self._audio_featurizer = AudioFeaturizer(feature_method=feature_method, target_sample_rate=target_sample_rate, target_dB=target_dB)
        self._text_featurizer = TextFeaturizer(vocab_filepath)

    def featurize(self, speech_segment):
        """提取语音片段的特征

        1. For audio parts, extract the audio features.
        2. For transcript parts, keep the original text or convert text string
           to a list of token indices in char-level.

        :param audio_segment: Speech segment to extract features from.
        :type audio_segment: SpeechSegment
        :return: A tuple of 1) spectrogram audio feature in 2darray, 2) list of
                 char-level token indices.
        :rtype: tuple
        """
        audio_feature = self._audio_featurizer.featurize(speech_segment)
        text_ids = self._text_featurizer.featurize(speech_segment.transcript)
        return audio_feature, text_ids

    @property
    def vocab_size(self):
        """返回词汇表大小

        :return: Vocabulary size.
        :rtype: int
        """
        return self._text_featurizer.vocab_size

    @property
    def vocab_list(self):
        """返回词汇表的list

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._text_featurizer.vocab_list

    @property
    def feature_dim(self):
        """返回词汇表大小

        :return: 词汇表大小
        :rtype: int
        """
        return self._audio_featurizer.feature_dim
