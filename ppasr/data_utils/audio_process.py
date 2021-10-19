from ppasr.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.featurizer.text_featurizer import TextFeaturizer
from ppasr.data_utils.audio import AudioSegment


class AudioProcess(object):
    """
    识别程序所使用的是对音频预处理的工具

    :param vocab_filepath: 词汇表的文件路径
    :type vocab_filepath: str
    """

    def __init__(self, vocab_filepath):
        self._audio_featurizer = AudioFeaturizer()
        self._text_featurizer = TextFeaturizer(vocab_filepath)

    def process_utterance(self, audio_file):
        """对语音数据加载、预处理

        :param audio_file: 音频文件的文件路径
        :type audio_file: str
        :return: 预处理的音频数据
        :rtype: 2darray
        """
        audio_segment = AudioSegment.from_file(audio_file)
        specgram = self._audio_featurizer.featurize(audio_segment)
        return specgram

    @property
    def vocab_list(self):
        """返回词汇表的list

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._text_featurizer.vocab_list
