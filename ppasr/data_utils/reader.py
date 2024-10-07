import json
from typing import List, Dict

import numpy as np
import paddle
from paddle.io import Dataset
from yeaudio.audio import AudioSegment
from yeaudio.augmentation import ReverbPerturbAugmentor, SpecAugmentor, SpecSubAugmentor
from yeaudio.augmentation import SpeedPerturbAugmentor, VolumePerturbAugmentor, NoisePerturbAugmentor

from ppasr.data_utils.audio_featurizer import AudioFeaturizer
from ppasr.data_utils.binary import DatasetReader
from ppasr.data_utils.tokenizer import PPASRTokenizer


# 音频数据加载器
class PPASRDataset(Dataset):
    """数据读取

    :param data_manifest: 数据列表或者数据列表文件路径
    :type data_manifest: str or List
    :param audio_featurizer: 音频特征化器
    :type audio_featurizer: AudioFeaturizer
    :param tokenizer: 文本分词器
    :type tokenizer: MASRTokenizer
    :param min_duration: 过滤的最小时长
    :type min_duration: float
    :param max_duration: 过滤的最大时长
    :type max_duration: float
    :param aug_conf: 数据增强配置
    :type aug_conf: Any
    :param manifest_type: 数据列表类型，只有文本格式和二进制格式两种
    :type manifest_type: str
    :param sample_rate: 采样率
    :type sample_rate: int
    :param use_dB_normalization: 是否使用dB归一化
    :type use_dB_normalization: bool
    :param target_dB: 归一化的目标音量
    :type target_dB: int
    :param mode: 数据模式，可选train、eval、test
    :type mode: str
    """
    def __init__(self,
                 data_manifest: [str or List],
                 audio_featurizer: AudioFeaturizer,
                 tokenizer: PPASRTokenizer = None,
                 min_duration=0,
                 max_duration=20,
                 aug_conf=None,
                 manifest_type='txt',
                 sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 mode="train"):
        super(PPASRDataset, self).__init__()
        assert manifest_type in ['txt', 'binary'], "数据列表类型只支持txt和binary"
        assert mode in ['train', 'eval', 'test'], "数据模式只支持train、val和test"
        self._audio_featurizer = audio_featurizer
        self._tokenizer = tokenizer
        self.manifest_type = manifest_type
        self._target_sample_rate = sample_rate
        self._use_dB_normalization = use_dB_normalization
        self._target_dB = target_dB
        self.mode = mode
        self.dataset_reader = None
        self.speed_augment = None
        self.volume_augment = None
        self.noise_augment = None
        self.reverb_augment = None
        self.spec_augment = None
        self.spec_sub_augment = None
        # 获取数据列表
        self.data_list = self.get_data_list(data_manifest, min_duration, max_duration)
        # 获取数据增强器
        if mode == "train" and aug_conf is not None:
            self.get_augmentor(aug_conf)

    def __getitem__(self, idx):
        data_list = self.get_one_list(idx)
        # 分割音频路径和标签
        audio_file, transcript = data_list["audio_filepath"], data_list["text"]
        # 如果后缀名为.npy的文件，那么直接读取
        if audio_file.endswith('.npy'):
            start_frame, end_frame = data_list["start_frame"], data_list["end_frame"]
            feature = np.load(audio_file)
            feature = feature[start_frame:end_frame, :]
            feature = paddle.to_tensor(feature, dtype=paddle.float32)
        else:
            if 'start_time' not in data_list.keys():
                # 读取音频
                audio_segment = AudioSegment.from_file(audio_file)
            else:
                start_time, end_time = data_list["start_time"], data_list["end_time"]
                # 分割读取音频
                audio_segment = AudioSegment.slice_from_file(audio_file, start=start_time, end=end_time)
            # 音频增强
            if self.mode == 'train':
                audio_segment = self.augment_audio(audio_segment)
            # 重采样
            if audio_segment.sample_rate != self._target_sample_rate:
                audio_segment.resample(self._target_sample_rate)
            # 音量归一化
            if self._use_dB_normalization:
                audio_segment.normalize(target_db=self._target_dB)
            # 预处理，提取特征
            feature = self._audio_featurizer.featurize(waveform=audio_segment.samples,
                                                       sample_rate=audio_segment.sample_rate)
        # 特征增强
        if self.mode == 'train':
            if isinstance(feature, paddle.Tensor):
                feature = feature.cpu().numpy()
            if self.spec_augment is not None:
                feature = self.spec_augment(feature)
            if self.spec_sub_augment is not None:
                feature = self.spec_sub_augment(feature)
            feature = paddle.to_tensor(feature, dtype=paddle.float32)
        # 有些任务值需要音频特征
        if self._tokenizer is None:
            return feature
        # 把文本标签转成token
        text_ids = self._tokenizer.text2ids(transcript)
        text_ids = paddle.to_tensor(text_ids, dtype=paddle.int32)
        return feature, text_ids

    def __len__(self):
        return len(self.data_list)

    # 获取数据列表
    def get_data_list(self, data_manifest, min_duration=0, max_duration=20) -> List[Dict]:
        """
        :param data_manifest: 数据列表或者数据列表文件路径
        :type data_manifest: str or List
        :param min_duration: 过滤的最小时长
        :type min_duration: float
        :param max_duration: 过滤的最大时长
        :type max_duration: float
        :return: 数据列表
        :rtype: List[dict]
        """
        data_list = []
        if isinstance(data_manifest, str):
            if self.manifest_type == 'txt':
                # 获取文本格式数据列表
                with open(data_manifest, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    # 跳过超出长度限制的音频
                    if line["duration"] < min_duration:
                        continue
                    if max_duration != -1 and line["duration"] > max_duration:
                        continue
                    data_list.append(dict(line))
            else:
                # 获取二进制的数据列表
                self.dataset_reader = DatasetReader(data_path=data_manifest,
                                                    min_duration=min_duration,
                                                    max_duration=max_duration)
                data_list = self.dataset_reader.get_keys()
        else:
            data_list = data_manifest
        return data_list

    # 获取数据列表中的一条数据
    def get_one_list(self, idx):
        if self.manifest_type == 'txt':
            data_list = self.data_list[idx]
        elif self.manifest_type == 'binary':
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            raise Exception(f'没有该类型：{self.manifest_type}')
        return data_list

    # 获取数据增强器
    def get_augmentor(self, aug_conf):
        if aug_conf.speed is not None:
            self.speed_augment = SpeedPerturbAugmentor(**aug_conf.speed)
        if aug_conf.volume is not None:
            self.volume_augment = VolumePerturbAugmentor(**aug_conf.volume)
        if aug_conf.noise is not None:
            self.noise_augment = NoisePerturbAugmentor(**aug_conf.noise)
        if aug_conf.reverb is not None:
            self.reverb_augment = ReverbPerturbAugmentor(**aug_conf.reverb)
        if aug_conf.spec_aug is not None:
            self.spec_augment = SpecAugmentor(**aug_conf.spec_aug)
        if aug_conf.spec_sub_aug is not None:
            self.spec_sub_augment = SpecSubAugmentor(**aug_conf.spec_sub_aug)

    # 音频增强
    def augment_audio(self, audio_segment):
        if self.speed_augment is not None:
            audio_segment = self.speed_augment(audio_segment)
        if self.volume_augment is not None:
            audio_segment = self.volume_augment(audio_segment)
        if self.noise_augment is not None:
            audio_segment = self.noise_augment(audio_segment)
        if self.reverb_augment is not None:
            audio_segment = self.reverb_augment(audio_segment)
        return audio_segment
