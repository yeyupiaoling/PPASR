"""查找最优的集束搜索方法的alpha参数和beta参数"""
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import numpy as np
import argparse
import functools
from ppasr.decoders.beam_search_decoder import BeamSearchDecoder
import paddle
from paddle.io import DataLoader
from tqdm import tqdm

from ppasr.utils.utils import add_arguments, print_arguments
from ppasr.data_utils.reader import PPASRDataset
from ppasr.data_utils.collate_fn import collate_fn
from ppasr.model_utils.deepspeech2.model import DeepSpeech2Model
from ppasr.utils.metrics import cer, wer
from ppasr.utils.utils import labels_to_string


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_data',         int,    -1,    "用于评估的数据数量，当为-1时使用全部数据")
add_arg('batch_size',       int,    16,    "评估是每一批数据的大小")
add_arg('beam_size',        int,    300,   "定向搜索的大小，范围建议:[5, 500]")
add_arg('num_proc_bsearch', int,    10,    "定向搜索方法使用CPU数量")
add_arg('num_alphas',       int,    30,    "用于调优的alpha候选项")
add_arg('num_betas',        int,    20,    "用于调优的beta候选项")
add_arg('alpha_from',       float,  1.0,   "alpha调优开始大小")
add_arg('alpha_to',         float,  3.2,   "alpha调优结速大小")
add_arg('beta_from',        float,  0.1,   "beta调优开始大小")
add_arg('beta_to',          float,  4.5,   "beta调优结速大小")
add_arg('cutoff_prob',      float,  0.99,  "剪枝的概率")
add_arg('cutoff_top_n',     int,    40,    "剪枝的最大值")
add_arg('use_model',        str,   'deepspeech2',             '所使用的模型')
add_arg('test_manifest',    str,   'dataset/manifest.test',   '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,   'dataset/vocabulary.txt',  '数据字典的路径')
add_arg('mean_std_path',    str,   'dataset/mean_std.npz',    '数据集的均值和标准值的npy文件路径')
add_arg('resume_model',     str,   'models/deepspeech2/best_model/', '模型的路径')
add_arg('metrics_type',     str,    'cer',               '计算错误率方法', choices=['cer', 'wer'])
add_arg('feature_method',   str,    'linear',            '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
add_arg('lang_model_path',  str,   'lm/zh_giga.no_cna_cmn.prune01244.klm',        "语言模型文件路径")
args = parser.parse_args()
print_arguments(args)


def tune():
    # 逐步调整alphas参数和betas参数
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    # 获取测试数据
    test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab, args.mean_std_path,
                                feature_method=args.feature_method)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=collate_fn,
                             use_shared_memory=False)

    # 获取模型
    if args.use_model == 'deepspeech2':
        model = DeepSpeech2Model(feat_size=test_dataset.feature_dim, vocab_size=test_dataset.vocab_size)
    else:
        raise Exception('没有该模型：%s' % args.use_model)

    assert os.path.exists(os.path.join(args.resume_model, 'model.pdparams')), "模型不存在！"
    model.set_state_dict(paddle.load(os.path.join(args.resume_model, 'model.pdparams')))
    model.eval()

    # 创建用于搜索的alphas参数和betas参数
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(round(alpha, 2), round(beta, 2)) for alpha in cand_alphas for beta in cand_betas]

    outputs = []
    labels = []
    # 多批增量调优参数
    print('开始识别数据...')
    used_sum = 0
    for inputs, label, input_lens, _ in tqdm(test_loader()):
        used_sum += inputs.shape[0]
        # 执行识别
        outs, _ = model(inputs, input_lens)
        outs = paddle.nn.functional.softmax(outs, 2)
        outputs.append(outs.numpy())
        labels.append(label.numpy())
        if args.num_data != -1 and used_sum >= args.num_data:break

    print('开始使用识别结果解码...')
    print('解码alpha和beta的排列：%s' % params_grid)
    # 搜索alphas参数和betas参数
    best_alpha, best_beta, best_cer = 0, 0, 1
    for i, (alpha, beta) in enumerate(params_grid):
        beam_search_decoder = BeamSearchDecoder(alpha, beta, args.lang_model_path, test_dataset.vocab_list)

        c = []
        print('正在解码[%d/%d]: (%.2f, %.2f)' % (i, len(params_grid), alpha, beta))
        for j in tqdm(range(len(labels))):
            outs, label = outputs[j], labels[j]
            out_strings = beam_search_decoder.decode_batch_beam_search(probs_split=outs,
                                                                       beam_alpha=alpha,
                                                                       beam_beta=beta,
                                                                       beam_size=args.beam_size,
                                                                       cutoff_prob=args.cutoff_prob,
                                                                       cutoff_top_n=args.cutoff_top_n,
                                                                       vocab_list=test_dataset.vocab_list,
                                                                       num_processes=args.num_proc_bsearch)
            labels_str = labels_to_string(label, test_dataset.vocab_list)
            for out_string, label in zip(*(out_strings, labels_str)):
                # 计算字错率或者词错率
                if args.metrics_type == 'wer':
                    c.append(wer(out_string, label))
                else:
                    c.append(cer(out_string, label))
        c = float(sum(c) / len(c))
        if c < best_cer:
            best_alpha = alpha
            best_beta = beta
            best_cer = c
        print('当alpha为：%f, beta为：%f，%s：%f' % (alpha, beta, args.metrics_type, c))
    print('【最后结果】当alpha为：%f, beta为：%f，%s最低，为：%f' % (best_alpha, best_beta, args.metrics_type, best_cer))


if __name__ == '__main__':
    tune()