"""特征归一化"""
import math

import numpy as np
import random
from tqdm import tqdm
from data_utils.utils import read_manifest
from data_utils.audio_featurizer import AudioFeaturizer


class FeatureNormalizer(object):
    """音频特征归一化类

    如果mean_std_filepath不是None，则normalizer将直接从文件初始化。否则，使用manifest_path应该给特征mean和stddev计算

    :param mean_std_filepath: 均值和标准值的文件路径
    :type mean_std_filepath: None|str
    :param manifest_path: 用于计算均值和标准值的数据列表，一般是训练的数据列表
    :type meanifest_path: None|str
    :param num_samples: 用于计算均值和标准值的音频数量
    :type num_samples: int
    :param random_seed: 随机种子
    :type random_seed: int
    :raises ValueError: 如果mean_std_filepath和manifest_path(或mean_std_filepath和featurize_func)都为None
    """

    def __init__(self,
                 mean_std_filepath,
                 manifest_path=None,
                 num_samples=5000,
                 random_seed=0):
        if not mean_std_filepath:
            if not manifest_path:
                raise ValueError("如果mean_std_filepath是None，那么meanifest_path和featurize_func不应该是None")
            self._rng = random.Random(random_seed)
            self.audio_featurizer = AudioFeaturizer()
            self._compute_mean_std(manifest_path, num_samples)
        else:
            self._read_mean_std_from_file(mean_std_filepath)

    def apply(self, features, eps=1e-20):
        """使用均值和标准值计算音频特征的归一化值

        :param features: 需要归一化的音频
        :type features: ndarray
        :param eps:  添加到标准值以提供数值稳定性
        :type eps: float
        :return: 已经归一化的数据
        :rtype: ndarray
        """
        return (features - self._mean) / (self._std + eps)

    def write_to_file(self, filepath):
        """将计算得到的均值和标准值写入到文件中

        :param filepath: 均值和标准值写入的文件路径
        :type filepath: str
        """
        np.savez(filepath, mean=self._mean, std=self._std)

    def _read_mean_std_from_file(self, filepath):
        """从文件中加载均值和标准值"""
        npzfile = np.load(filepath)
        self._mean = npzfile["mean"]
        self._std = npzfile["std"]

    def _compute_mean_std(self, manifest_path, num_samples):
        """从随机抽样的实例中计算均值和标准值"""
        manifest = read_manifest(manifest_path)
        if num_samples < 0:
            sampled_manifest = manifest
        else:
            sampled_manifest = self._rng.sample(manifest, num_samples)
        # 求总和
        std, means = None, None
        number = 0
        for instance in tqdm(sampled_manifest):
            audio = self.audio_featurizer.load_audio_file(instance["audio_path"])
            feature = self.audio_featurizer.featurize(audio)
            number += feature.shape[1]
            sums = np.sum(feature, axis=1)
            if means is None:
                means = sums
            else:
                means += sums
            square_sums = np.sum(np.square(feature), axis=1)
            if std is None:
                std = square_sums
            else:
                std += square_sums
        # 求总和的均值和标准值
        for i in range(len(means)):
            means[i] /= number
            std[i] = std[i] / number - means[i] * means[i]
            if std[i] < 1.0e-20:
                std[i] = 1.0e-20
            std[i] = math.sqrt(std[i])
        self._mean = means.reshape([-1, 1])
        self._std = std.reshape([-1, 1])
