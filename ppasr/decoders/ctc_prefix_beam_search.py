from collections import defaultdict
from typing import List

import paddle

from ppasr.decoders.utils import log_add


def ctc_prefix_beam_search(
        ctc_probs: paddle.Tensor,
        ctc_lens: paddle.Tensor,
        beam_size: int = 10,
        blank_id: int = 0,
) -> [List, List]:
    """CTC prefix beam search

    param ctc_probs: (B, maxlen, vocab_size) 模型编码器输出的概率分布
    param ctc_lens: (B, ) 每个样本的实际长度
    param beam_size: 解码搜索大小
    param blank_id: 空白标签的id
    return: 解码结果，和所有解码结果，用于attention_rescoring解码器使用
    """
    batch_size = ctc_probs.shape[0]
    results, hyps_list = [], []
    # CTC prefix beam search can not be paralleled, so search one by one
    for i in range(batch_size):
        ctc_prob = ctc_probs[i]
        num_t = ctc_lens[i]
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        # blank_ending_score and  none_blank_ending_score in ln domain
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, num_t):
            logp = ctc_prob[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == blank_id:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(), key=lambda x: log_add(list(x[1])), reverse=True)
            cur_hyps = next_hyps[:beam_size]

        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        results.append(list(hyps[0][0]))
        hyps_list.append(hyps)
    return results, hyps_list
