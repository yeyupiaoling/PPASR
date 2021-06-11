import numpy as np


class AudioTool(object):
    def __init__(self):
        pass

    def normalize(self, audio, target_db=-20, max_gain_db=300.0):
        """将音频归一化，使其具有所需的有效值(以分贝为单位)

        :param audio: 音频数据
        :type audio numpy
        :param target_db: 目标均方根值(分贝)。这个值应该小于0.0，因为0.0是全尺寸音频
        :type target_db: float
        :param max_gain_db: 可用于归一化的dB最大增益量。这是为了防止nans试图规范化一个全为0的信号
        :type max_gain_db: float
        :raises ValueError: 如果将段归一化到target_db值所需的增益超过max_gain_db
        """
        gain = target_db - self.rms_db(audio)
        if gain > max_gain_db:
            raise ValueError("无法将段规范化到 %f dB，增益已经超过max_gain_db (%f dB)" % (target_db, max_gain_db))
        return self.gain_db(audio, min(max_gain_db, target_db - self.rms_db(audio)))

    @staticmethod
    def gain_db(audio, gain):
        """对音频施加分贝增益。

        :param audio: 音频数据
        :type audio numpy
        :param gain: 适用于音频的分贝增益
        :type gain: float|1darray
        """
        audio *= 10. ** (gain / 20.)
        return audio

    @staticmethod
    def rms_db(audio):
        """返回以分贝为单位的音频均方根能量

        :return: 以分贝表示的均方根能量
        :rtype: float
        """
        mean_square = np.mean(audio ** 2)
        return 10 * np.log10(mean_square)
