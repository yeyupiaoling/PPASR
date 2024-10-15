import multiprocessing
import platform
from collections import defaultdict
from typing import List

import numpy as np
import paddle

from ppasr.decoders.utils import log_add


# 多进行推理需要用到的
def run_ctc_prefix_beam_search(ctc_prob: List,
                               num_t: int,
                               beam_size: int = 10,
                               blank_id: int = 0):
    ctc_prob = np.array(ctc_prob, dtype=np.float32)
    # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
    # blank_ending_score and  none_blank_ending_score in ln domain
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, num_t):
        logp = ctc_prob[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        sorted_indices = np.argsort(logp)[::-1]  # 从大到小排序
        top_k_index = sorted_indices[:beam_size]  # (beam_size,)
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
    return list(hyps[0][0]), hyps


def ctc_prefix_beam_search(ctc_probs: paddle.Tensor,
                           ctc_lens: paddle.Tensor,
                           num_workers: int = 4,
                           beam_size: int = 10,
                           blank_id: int = 0) -> [List, List]:
    """CTC prefix beam search

    param ctc_probs: (B, maxlen, vocab_size) 模型编码器输出的概率分布
    param ctc_lens: (B, ) 每个样本的实际长度
    param num_workers: 并行解码的进程数
    param beam_size: 解码搜索大小
    param blank_id: 空白标签的id
    return: 解码结果，和所有解码结果，用于attention_rescoring解码器使用
    """
    # 如果只有一条数据，直接解码
    batch_size = ctc_probs.shape[0]
    if batch_size == 1:
        ctc_prob = ctc_probs[0].tolist()
        num_t = ctc_lens[0].item()
        result, hyps = run_ctc_prefix_beam_search(ctc_prob, num_t, beam_size, blank_id)
        return [result], [hyps]
    # Windows系统不支持多进程
    if platform.system() == 'Windows':
        results, hyps_list = [], []
        for i in range(batch_size):
            ctc_prob = ctc_probs[i].tolist()
            num_t = ctc_lens[i].item()
            result, hyps = run_ctc_prefix_beam_search(ctc_prob, num_t, beam_size, blank_id)
            results.append(result)
            hyps_list.append(hyps)
        return results, hyps_list
    # 其他系统使用多进程并行解码
    num_processes = min(batch_size, num_workers)
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    processes_results = []
    for i in range(batch_size):
        ctc_prob = ctc_probs[i].tolist()
        num_t = ctc_lens[i].item()
        args = (ctc_prob, num_t, beam_size, blank_id)
        processes_results.append(pool.apply_async(run_ctc_prefix_beam_search, args))
    pool.close()
    pool.join()

    # 获取每个进程的结果
    results, hyps_list = [], []
    for result in processes_results:
        r = result.get()
        results.append(r[0])
        hyps_list.append(r[1])
    return results, hyps_list
