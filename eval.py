import argparse
import functools
import os
import time

import numpy as np
import paddle
from paddle.io import DataLoader
from model_utils.deepspeech2 import DeepSpeech2Model
from tqdm import tqdm
from data.utility import add_arguments, print_arguments
from data_utils.reader import PPASRDataset, collate_fn
from utils.decoder import GreedyDecoder


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,   32,                        '训练的批量大小')
add_arg('num_workers',      int,   8,                         '读取数据的线程数量')
add_arg('num_conv_layers',  int,   2,                         '卷积层数量')
add_arg('num_rnn_layers',   int,   3,                         '循环神经网络的数量')
add_arg('rnn_layer_size',   int,   1024,                      '循环神经网络的大小')
add_arg('test_manifest',    str,   'dataset/manifest.test',   '测试数据的数据列表路径')
add_arg('dataset_vocab',    str,   'dataset/vocabulary.json', '数据字典的路径')
add_arg('mean_std_path',    str,   'dataset/mean_std.npz',    '数据集的均值和标准值的npy文件路径')
add_arg('model_path',       str,   'models/step_final/',      '模型的路径')
args = parser.parse_args()


print_arguments(args)
# 获取测试数据
test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab, args.mean_std_path)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         collate_fn=collate_fn,
                         num_workers=args.num_workers,
                         use_shared_memory=False)
# 获取解码器，用于评估
greedy_decoder = GreedyDecoder(test_dataset.vocabulary)
# 获取模型
model = DeepSpeech2Model(feat_size=test_dataset.feature_dim,
                         dict_size=len(test_dataset.vocabulary),
                         num_conv_layers=args.num_conv_layers,
                         num_rnn_layers=args.num_rnn_layers,
                         rnn_size=args.rnn_layer_size)
model.set_state_dict(paddle.load(os.path.join(args.model_path, 'model.pdparams')))
model.eval()


# 评估模型
@paddle.no_grad()
def evaluate():
    cer = []
    for batch_id, (inputs, labels, input_lens, _) in enumerate(tqdm(test_loader())):
        # 执行识别
        outs, _ = model(inputs, input_lens)
        outs = paddle.nn.functional.softmax(outs, 2)
        # 解码获取识别结果
        out_strings, out_offsets = greedy_decoder.decode(outs)
        labels = greedy_decoder.convert_to_strings(labels)
        for out_string, label in zip(*(out_strings, labels)):
            # 计算字错率
            c = greedy_decoder.cer(out_string[0], label[0]) / float(len(label[0]))
            cer.append(c)
    cer = float(np.mean(cer))
    return cer


if __name__ == '__main__':
    start = time.time()
    cer = evaluate()
    end = time.time()
    print('识别时间：%dms，字错率：%f' % (round((end - start) * 1000), cer))
