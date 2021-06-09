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
    max_prob_list = [probs_seq[i][max_index_list[i]] for i in range(len(max_index_list)) if max_index_list[i] != blank_index]
    # 删除连续的重复索引和空索引
    index_list = [index_group[0] for index_group in groupby(max_index_list)]
    index_list = [index for index in index_list if index != blank_index]
    # 索引列表转换为字符串
    text = ''.join([vocabulary[index] for index in index_list])
    score = 0
    if len(max_prob_list) > 0:
        score = float(sum(max_prob_list) / len(max_prob_list)) * 100.0
    return score, text


def greedy_decoder_batch(probs_split, vocabulary):
    """CTC贪婪(最佳路径)解码器
    :param probs_split: 一批包含2D的概率表
    :type probs_split: list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :return: 字符串列表
    :rtype: list
    """
    results = []
    for i, probs in enumerate(probs_split):
        output_transcription = greedy_decoder(probs, vocabulary)
        results.append(output_transcription[1])
    return results
