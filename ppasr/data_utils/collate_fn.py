import random

import numpy as np


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch_sorted = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    freq_size = batch_sorted[0][0].shape[1]
    max_audio_length = batch_sorted[0][0].shape[0]
    batch_size = len(batch_sorted)
    # 找出标签最长的
    batch_temp = sorted(batch_sorted, key=lambda sample: len(sample[1]), reverse=True)
    max_label_length = len(batch_temp[0][1])
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype=np.float32)
    labels = np.ones((batch_size, max_label_length), dtype=np.int32) * -1
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch_sorted[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[0]
        label_length = target.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length, :] = tensor[:, :]
        labels[x, :label_length] = target[:]
        input_lens.append(seq_length)
        label_lens.append(label_length)
    input_lens = np.array(input_lens, dtype=np.int64)
    label_lens = np.array(label_lens, dtype=np.int64)
    # 打乱数据
    indices = np.arange(batch_size).tolist()
    random.shuffle(indices)
    inputs = inputs[indices]
    labels = labels[indices]
    input_lens = input_lens[indices]
    label_lens = label_lens[indices]
    return inputs, labels, input_lens, label_lens
