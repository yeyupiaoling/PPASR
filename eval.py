import argparse
import functools
import os
import time

import paddle
from paddle.io import DataLoader
from tqdm import tqdm

from data.utility import add_arguments, print_arguments
from data_utils.reader import PPASRDataset, collate_fn
from decoders.ctc_greedy_decoder import greedy_decoder_batch
from model_utils.deepspeech2 import DeepSpeech2Model
from utils.metrics import cer
from utils.utils import labels_to_string

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    32,                       '训练的批量大小')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('num_conv_layers',  int,    2,                        '卷积层数量')
add_arg('num_rnn_layers',   int,    3,                        '循环神经网络的数量')
add_arg('rnn_layer_size',   int,    1024,                     '循环神经网络的大小')
add_arg('alpha',            float,  1.2,                      '集束搜索的LM系数')
add_arg('beta',             float,  0.35,                     '集束搜索的WC系数')
add_arg('beam_size',        int,    10,                       '集束搜索的大小，范围:[5, 500]')
add_arg('num_proc_bsearch', int,    8,                        '集束搜索方法使用CPU数量')
add_arg('cutoff_prob',      float,  1.0,                      '剪枝的概率')
add_arg('cutoff_top_n',     int,    40,                       '剪枝的最大值')
add_arg('test_manifest',    str,   'dataset/manifest.test',   '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,   'dataset/vocabulary.json', '数据字典的路径')
add_arg('mean_std_path',    str,   'dataset/mean_std.npz',    '数据集的均值和标准值的npy文件路径')
add_arg('model_path',       str,   'models/step_final/',      '模型的路径')
add_arg('decoder',          str,   'ctc_beam_search',         '结果解码方法', choices=['ctc_beam_search', 'ctc_greedy'])
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',        "语言模型文件路径")
args = parser.parse_args()


print_arguments(args)
# 获取测试数据
test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab, args.mean_std_path)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         collate_fn=collate_fn,
                         num_workers=args.num_workers,
                         use_shared_memory=False)

# 获取模型
model = DeepSpeech2Model(feat_size=test_dataset.feature_dim,
                         dict_size=len(test_dataset.vocabulary),
                         num_conv_layers=args.num_conv_layers,
                         num_rnn_layers=args.num_rnn_layers,
                         rnn_size=args.rnn_layer_size)
model.set_state_dict(paddle.load(os.path.join(args.model_path, 'model.pdparams')))
model.eval()

# 集束搜索方法的处理
if args.decoder == "ctc_beam_search":
    try:
        from decoders.beam_search_decoder import BeamSearchDecoder
        beam_search_decoder = BeamSearchDecoder(args.alpha, args.beta, args.lang_model_path, test_dataset.vocabulary)
    except ModuleNotFoundError:
        raise Exception('缺少ctc_decoders库，请在decoders目录中执行setup.sh编译，如果是Windows系统，请使用ctc_greedy。')


# 执行解码
def decoder(outs, vocabulary):
    if args.decoder == 'ctc_greedy':
        result = greedy_decoder_batch(outs, vocabulary)
    else:
        result = beam_search_decoder.decode_batch_beam_search(probs_split=outs,
                                                              beam_alpha=args.alpha,
                                                              beam_beta=args.beta,
                                                              beam_size=args.beam_size,
                                                              cutoff_prob=args.cutoff_prob,
                                                              cutoff_top_n=args.cutoff_top_n,
                                                              vocab_list=test_dataset.vocabulary,
                                                              num_processes=args.num_proc_bsearch)
    return result


# 评估模型
@paddle.no_grad()
def evaluate():
    c = []
    for inputs, labels, input_lens, _ in tqdm(test_loader()):
        # 执行识别
        outs, _ = model(inputs, input_lens)
        outs = paddle.nn.functional.softmax(outs, 2)
        # 解码获取识别结果
        out_strings = decoder(outs.numpy(), test_dataset.vocabulary)
        labels_str = labels_to_string(labels.numpy(), test_dataset.vocabulary)
        for out_string, label in zip(*(out_strings, labels_str)):
            # 计算字错率
            c.append(cer(out_string, label) / float(len(label)))
    c = float(sum(c) / len(c))
    return c


if __name__ == '__main__':
    start = time.time()
    cer = evaluate()
    end = time.time()
    print('评估消耗时间：%ds，字错率：%f' % ((end - start), cer))
