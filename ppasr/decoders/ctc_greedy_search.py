from typing import List

import paddle

from ppasr.decoders.utils import remove_duplicates_and_blank
from ppasr.model_utils.utils.mask import make_pad_mask


def ctc_greedy_search(ctc_probs: paddle.Tensor,
                      ctc_lens: paddle.Tensor,
                      blank_id: int = 0) -> List:
    """贪心解码器

    param ctc_probs: (B, maxlen, vocab_size) 模型编码器输出的概率分布
    param ctc_lens: (B, ) 每个样本的实际长度
    param blank_id: 空白标签的id
    return: 解码结果
    """
    batch_size = ctc_probs.shape[0]
    maxlen = ctc_probs.shape[1]
    topk_prob, topk_index = ctc_probs.topk(1, axis=2)  # (B, maxlen, 1)
    topk_index = topk_index.view((batch_size, maxlen))  # (B, maxlen)
    mask = make_pad_mask(ctc_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, blank_id)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores, _ = topk_prob.max(1)
    scores = scores.cpu().detach().numpy().tolist()
    results = []
    for hyp, score in zip(hyps, scores):
        r = remove_duplicates_and_blank(hyp, blank_id)
        results.append(r)
    return results
