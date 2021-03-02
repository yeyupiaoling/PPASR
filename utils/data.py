import wave

import librosa
import numpy as np
from paddle.io import Dataset


# 加载二进制音频文件，转成numpy值
def load_audio(wav_path, normalize=True):
    with wave.open(wav_path) as wav:
        wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16")
        wav = wav.astype("float")
    if normalize:
        wav = (wav - wav.mean()) / wav.std()
    return wav


# 把音频数据执行短时傅里叶变换
def audio_to_stft(wav, normalize=True):
    D = librosa.stft(wav, n_fft=320, hop_length=160, win_length=320, window="hamming")
    spec, phase = librosa.magphase(D)
    spec = np.log1p(spec)
    if normalize:
        spec = (spec - spec.mean()) / spec.std()
    return spec


# 音频数据加载器
class PPASRDataset(Dataset):
    def __init__(self, data_list, dict_path):
        super(PPASRDataset, self).__init__()
        # 获取数据列表
        with open(data_list) as f:
            idx = f.readlines()
        self.idx = [x.strip().split(",") for x in idx]
        # 加载数据字典
        with open(dict_path) as f:
            labels = eval(f.read())
        self.vocabulary = dict([(labels[i], i) for i in range(len(labels))])

    def __getitem__(self, idx):
        # 分割音频路径和标签
        wav_path, _, transcript = self.idx[idx]
        # 读取音频并转换为短时傅里叶变换
        wav = load_audio(wav_path)
        stft = audio_to_stft(wav)
        # 将字符标签转换为int数据
        transcript = list(filter(None, [self.vocabulary.get(x) for x in transcript]))
        transcript = np.array(transcript, dtype='int32')
        return stft, transcript

    def __len__(self):
        return len(self.idx)


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    freq_size = batch[0][0].shape[0]
    max_audio_length = batch[0][0].shape[1]
    batch_size = len(batch)
    # 找出标签最长的
    batch_temp = sorted(batch, key=lambda sample: len(sample[1]), reverse=True)
    max_label_length = len(batch_temp[0][1])
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, freq_size, max_audio_length), dtype='float32')
    labels = np.zeros((batch_size, max_label_length), dtype='int32')
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[1]
        label_length = target.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :seq_length] = tensor[:, :]
        labels[x, :label_length] = target[:]
        input_lens.append(seq_length)
        label_lens.append(len(target))
    input_lens = np.array(input_lens, dtype='int64')
    label_lens = np.array(label_lens, dtype='int64')
    return inputs, labels, input_lens, label_lens
