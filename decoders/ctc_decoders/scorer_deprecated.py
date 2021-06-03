"""External Scorer for Beam Search Decoder."""

import os
import kenlm
import numpy as np


class Scorer(object):
    """在集束搜索解码中对前缀或整句进行外部评分，包括n-gram语言模型的评分和单词计数

    :param alpha: 与语言模型相关的参数。当alpha = 0时不要使用语言模型
    :type alpha: float
    :param beta: 与字计数相关的参数。当beta = 0时不要使用统计字
    :type beta: float
    :model_path: 语言模型的路径
    :type model_path: str
    """

    def __init__(self, alpha, beta, model_path):
        self._alpha = alpha
        self._beta = beta
        if not os.path.isfile(model_path):
            raise IOError("Invaid language model path: %s" % model_path)
        self._language_model = kenlm.LanguageModel(model_path)

    # n-gram language model scoring
    def _language_model_score(self, sentence):
        # log10 prob of last word
        log_cond_prob = list(
            self._language_model.full_scores(sentence, eos=False))[-1][0]
        return np.power(10, log_cond_prob)

    # word insertion term
    def _word_count(self, sentence):
        words = sentence.strip().split(' ')
        return len(words)

    # reset alpha and beta
    def reset_params(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    # execute evaluation
    def __call__(self, sentence, log=False):
        """评价功能，收集所有不同的分数，并返回最后一个

        :param sentence: 输入语句进行计算
        :type sentence: str
        :param log: 是否以日志形式返回分数
        :type log: bool
        :return: 评价分数，用小数或对数表示
        :rtype: float
        """
        lm = self._language_model_score(sentence)
        word_cnt = self._word_count(sentence)
        if not log:
            score = np.power(lm, self._alpha) * np.power(word_cnt, self._beta)
        else:
            score = self._alpha * np.log(lm) + self._beta * np.log(word_cnt)
        return score
