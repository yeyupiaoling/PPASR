import numpy as np
from ppasr.data_utils.audio import AudioSegment


class SpeechSegment(AudioSegment):
    """语音片段抽象是音频片段的一个子类，附加文字记录。

    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: 训练数据的采样率
    :type sample_rate: int
    :param transcript: 音频文件对应的文本
    :type transript: str
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, transcript):
        AudioSegment.__init__(self, samples, sample_rate)
        self._transcript = transcript

    def __eq__(self, other):
        """Return whether two objects are equal.
        """
        if not AudioSegment.__eq__(self, other):
            return False
        if self._transcript != other._transcript:
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    @classmethod
    def from_file(cls, filepath, transcript):
        """从音频文件和相应的文本创建语音片段

        :param filepath: 音频文件路径
        :type filepath: str|file
        :param transcript: 音频文件对应的文本
        :type transript: str
        :return: Speech segment instance.
        :rtype: SpeechSegment
        """
        audio = AudioSegment.from_file(filepath)
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def from_bytes(cls, bytes, transcript):
        """从字节串和相应的文本创建语音片段

        :param bytes: 包含音频样本的字节字符串
        :type bytes: str
        :param transcript: 音频文件对应的文本
        :type transript: str
        :return: Speech segment instance.
        :rtype: Speech Segment
        """
        audio = AudioSegment.from_bytes(bytes)
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def concatenate(cls, *segments):
        """将任意数量的语音片段连接在一起，音频和文本都将被连接

        :param *segments: 要连接的输入语音片段
        :type *segments: tuple of SpeechSegment
        :return: 返回SpeechSegment实例
        :rtype: SpeechSegment
        :raises ValueError: 不能用不同的抽样率连接片段
        :raises TypeError: 只有相同类型SpeechSegment实例的语音片段可以连接
        """
        if len(segments) == 0:
            raise ValueError("音频片段为空")
        sample_rate = segments[0]._sample_rate
        transcripts = ""
        for seg in segments:
            if sample_rate != seg._sample_rate:
                raise ValueError("不能用不同的抽样率连接片段")
            if type(seg) is not cls:
                raise TypeError("只有相同类型SpeechSegment实例的语音片段可以连接")
            transcripts += seg._transcript
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate, transcripts)

    @classmethod
    def slice_from_file(cls, filepath, transcript, start=None, end=None):
        """只加载一小部分SpeechSegment，而不需要将整个文件加载到内存中，这是非常浪费的。

        :param filepath:文件路径或文件对象到音频文件
        :type filepath: str|file
        :param start: 开始时间，单位为秒。如果start是负的，则它从末尾开始计算。如果没有提供，这个函数将从最开始读取。
        :type start: float
        :param end: 结束时间，单位为秒。如果end是负的，则它从末尾开始计算。如果没有提供，默认的行为是读取到文件的末尾。
        :type end: float
        :param transcript: 音频文件对应的文本，如果没有提供，默认值是一个空字符串。
        :type transript: str
        :return: SpeechSegment实例
        :rtype: SpeechSegment
        """
        audio = AudioSegment.slice_from_file(filepath, start, end)
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """创建指定安静音频长度和采样率的SpeechSegment实例，音频文件对应的文本将为空字符串。

        :param duration: 安静音频的时间，单位秒
        :type duration: float
        :param sample_rate: 音频采样率
        :type sample_rate: float
        :return: 安静音频SpeechSegment实例
        :rtype: SpeechSegment
        """
        audio = AudioSegment.make_silence(duration, sample_rate)
        return cls(audio.samples, audio.sample_rate, "")

    @property
    def transcript(self):
        """返回音频文件对应的文本

        :return: 音频文件对应的文本
        :rtype: str
        """
        return self._transcript
