import json
import os
import random

import numpy as np
import paddle
from loguru import logger
from tqdm import tqdm
from paddle.io import DataLoader


from ppasr.data_utils.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.reader import PPASRDataset
from ppasr.data_utils.utils import read_manifest

__all__ = ['FeatureNormalizer']


class FeatureNormalizer(object):
    """音频特征归一化类

    :param mean_istd_filepath: 均值和标准值的文件路径
    """

    def __init__(self, mean_istd_filepath, eps=1e-20):
        self.mean_std_filepath = mean_istd_filepath
        # 读取归一化文件
        if os.path.exists(mean_istd_filepath):
            self.mean, self.istd = self._read_mean_istd_from_file(mean_istd_filepath)
            self.istd = np.maximum(self.istd, eps)

    @staticmethod
    def _read_mean_istd_from_file(filepath):
        """从文件中加载均值和标准值"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            mean = np.array(data["mean"], dtype=np.float32)
            istd = np.array(data["istd"], dtype=np.float32)
        return mean, istd

    def compute_mean_istd(self,
                          preprocess_conf,
                          manifest_path,
                          data_loader_conf,
                          num_samples=5000):
        """从随机抽样的实例中计算均值和标准值，并写入到文件中

        :param preprocess_conf: 数据预处理配置参数
        :param manifest_path: 数据列表文件路径
        :param data_loader_conf: DataLoader参数
        :param num_samples: 用于计算均值和标准值的音频数量
        """
        manifest = read_manifest(manifest_path, max_duration=30, min_duration=0.5)
        if num_samples < 0 or num_samples > len(manifest):
            sampled_manifest = manifest
        else:
            sampled_manifest = random.sample(manifest, num_samples)
        logger.info('开始抽取{}条数据计算均值和标准值...'.format(len(sampled_manifest)))
        audio_featurizer = AudioFeaturizer(**preprocess_conf)
        dataset = PPASRDataset(sampled_manifest, audio_featurizer=audio_featurizer)
        test_loader = DataLoader(dataset=dataset, collate_fn=collate_fn, **data_loader_conf)
        with paddle.no_grad():
            # 求总和
            std, means = None, None
            number = 0
            for std1, means1, number1 in tqdm(test_loader()):
                number += number1
                if means is None:
                    means = means1
                else:
                    means += means1
                if std is None:
                    std = std1
                else:
                    std += std1
            # 求总和的均值和标准值
            for i in range(len(means)):
                means[i] /= number
                std[i] = std[i] / number - means[i] * means[i]
                if std[i] < 1.0e-20:
                    std[i] = 1.0e-20
                std[i] = paddle.sqrt(std[i])
        istd = 1.0 / std
        # 写入到文件中
        data = {'mean': means.cpu().numpy().tolist(),
                'istd': istd.cpu().numpy().tolist(),
                'feature_method': preprocess_conf.feature_method}
        with open(self.mean_std_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)


def collate_fn(features):
    std, means = None, None
    number = 0
    for feature in features:
        number += feature.shape[0]
        sums = paddle.sum(feature, axis=0)
        if means is None:
            means = sums
        else:
            means += sums
        square_sums = paddle.sum(paddle.square(feature), axis=0)
        if std is None:
            std = square_sums
        else:
            std += square_sums
    return std, means, number
