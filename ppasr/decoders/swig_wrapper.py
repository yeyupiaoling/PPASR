import paddlespeech_ctcdecoders


class Scorer(paddlespeech_ctcdecoders.Scorer):
    """Wrapper for Scorer.

    :param alpha: 与语言模型相关的参数。当alpha = 0时不要使用语言模型
    :type alpha: float
    :param beta: 与字计数相关的参数。当beta = 0时不要使用统计字
    :type beta: float
    :model_path: 语言模型的路径
    :type model_path: str
    :param vocabulary: 词汇列表
    :type vocabulary: list
    """

    def __init__(self, alpha, beta, model_path, vocabulary):
        paddlespeech_ctcdecoders.Scorer.__init__(self, alpha, beta, model_path, vocabulary)


def ctc_greedy_decoding(probs_seq, vocabulary, blank_id):
    """CTC贪婪(最佳路径)解码器

    :param probs_seq: 每一条都是2D的概率表。每个元素都是浮点数
    :type probs_seq: 2-D list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :return:解码结果
    :rtype: str
    """
    result = paddlespeech_ctcdecoders.ctc_greedy_decoding(probs_seq.tolist(), vocabulary, blank_id)
    return result


def ctc_beam_search_decoding(probs_seq,
                             vocabulary,
                             beam_size,
                             cutoff_prob=1.0,
                             cutoff_top_n=40,
                             blank_id=0,
                             ext_scoring_func=None):
    """集束搜索解码器

    :param probs_seq: 单个2-D概率分布列表，每个元素是词汇表和空白上的标准化概率列表
    :type probs_seq: 2-D list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :param beam_size: 集束搜索宽度
    :type beam_size: int
    :param cutoff_prob: 剪枝中的截断概率，默认1.0，没有剪枝
    :type cutoff_prob: float
    :param cutoff_top_n: 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
    :type cutoff_top_n: int
    :param blank_id 空白索引
    :type blank_id int
    :param ext_scoring_func: 外部评分功能部分解码句子，如字计数或语言模型
    :type ext_scoring_func: callable
    :return: 解码结果为log概率和句子的元组列表，按概率降序排列
    :rtype: list
    """
    beam_results = paddlespeech_ctcdecoders.ctc_beam_search_decoding(
        probs_seq.tolist(), vocabulary, beam_size, cutoff_prob, cutoff_top_n, ext_scoring_func, blank_id)
    beam_results = [(res[0], res[1]) for res in beam_results]
    return beam_results


def ctc_beam_search_decoding_batch(probs_split,
                                   vocabulary,
                                   beam_size,
                                   num_processes,
                                   cutoff_prob=1.0,
                                   cutoff_top_n=40,
                                   blank_id=0,
                                   ext_scoring_func=None):
    """Wrapper for the batched CTC beam search decoder.

    :param probs_seq: 3-D列表，每个元素作为ctc_beam_search_decoder()使用的2-D概率列表的实例
    :type probs_seq: 3-D list
    :param vocabulary: 词汇列表
    :type vocabulary: list
    :param beam_size: 集束搜索宽度
    :type beam_size: int
    :param cutoff_prob: 剪枝中的截断概率，默认1.0，没有剪枝
    :type cutoff_prob: float
    :param cutoff_top_n: 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
    :type cutoff_top_n: int
    :param blank_id 空白索引
    :type blank_id int
    :param num_processes: 并行解码进程数
    :type num_processes: int
    :param ext_scoring_func: 外部评分功能部分解码句子，如字计数或语言模型
    :type ext_scoring_func: callable
    :return: 解码结果为log概率和句子的元组列表，按概率降序排列的列表
    :rtype: list
    """
    probs_split = [probs_seq.tolist() for probs_seq in probs_split]

    batch_beam_results = paddlespeech_ctcdecoders.ctc_beam_search_decoding_batch(
        probs_split, vocabulary, beam_size, num_processes, cutoff_prob,
        cutoff_top_n, ext_scoring_func, blank_id)
    batch_beam_results = [[(res[0], res[1]) for res in beam_results]
                          for beam_results in batch_beam_results]
    return batch_beam_results


class CTCBeamSearchDecoder(paddlespeech_ctcdecoders.CtcBeamSearchDecoderBatch):
    """Wrapper for CtcBeamSearchDecoderBatch.
    Args:
        vocab_list (list): 词汇列表
        beam_size (int): 集束搜索宽度
        num_processes (int): 并行解码进程数
        param cutoff_prob (float): 剪枝中的截断概率，默认1.0，没有剪枝
        cutoff_top_n (int): 剪枝时的截断数，仅在词汇表中具有最大probs的cutoff_top_n字符用于光束搜索，默认为40
        param ext_scorer (Scorer): 外部评分功能部分解码句子，如字计数或语言模型
    """

    def __init__(self, vocab_list, batch_size, beam_size, num_processes,
                 cutoff_prob, cutoff_top_n, _ext_scorer, blank_id):
        paddlespeech_ctcdecoders.CtcBeamSearchDecoderBatch.__init__(
            self, vocab_list, batch_size, beam_size, num_processes, cutoff_prob,
            cutoff_top_n, _ext_scorer, blank_id)
