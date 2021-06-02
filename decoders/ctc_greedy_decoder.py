from itertools import groupby

import numpy as np


def greedy_decoder(probs_seq, vocabulary, blank_index=0):
    """CTC贪婪(最佳路径)解码器

    由最可能的令牌组成的路径将被进一步后处理到去掉连续重复和所有空白

    :param probs_seq: 每一条都是2D的概率表。每个元素都是浮点数概率的列表一个字符
    :type probs_seq: list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :param blank_index 需要移除的空白索引
    :type blank_index int
    :return: 解码后得到的字符串
    :rtype: baseline
    """
    # 获得每个时间步的最佳索引
    max_index_list = list(np.array(probs_seq).argmax(axis=1))
    # 删除连续的重复索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    index_list = [index for index in index_list if index != blank_index]
    # 索引列表转换为字符串
    return ''.join([vocabulary[index] for index in index_list])


def greedy_decoder_batch(probs_split, vocabulary):
    """Decode by best path for a batch of probs matrix input.
    :param probs_split: List of 2-D probability matrix, and each consists
                        of prob vectors for one speech utterancce.
    :param probs_split: List of matrix
    :param vocab_list: List of tokens in the vocabulary, for decoding.
    :type vocab_list: list
    :return: List of transcription texts.
    :rtype: List of str
    """
    results = []
    for i, probs in enumerate(probs_split):
        output_transcription = greedy_decoder(probs, vocabulary)
        results.append(output_transcription)
    return results
