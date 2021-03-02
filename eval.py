import argparse
import functools
import os
import time

import numpy as np
import paddle
from paddle.io import DataLoader

from data.utility import add_arguments, print_arguments
from utils.data import PPASRDataset, collate_fn
from utils.decoder import GreedyDecoder
from utils.model import PPASR


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',     int,  32,                      '训练的批量大小')
add_arg('num_workers',    int,  8,                       '读取数据的线程数量')
add_arg('test_manifest',  str,  'dataset/manifest.test', '测试数据的数据列表路径')
add_arg('dataset_vocab',  str,  'dataset/zh_vocab.json', '数据字典的路径')
add_arg('model_path',     str,  'models/step_final/',    '模型的路径')
args = parser.parse_args()


print_arguments(args)
# 获取测试数据
test_dataset = PPASRDataset(args.test_manifest, args.dataset_vocab)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch_size,
                         collate_fn=collate_fn,
                         num_workers=args.num_workers)
# 获取解码器，用于评估
greedy_decoder = GreedyDecoder(test_dataset.vocabulary)
# 获取模型
model = PPASR(test_dataset.vocabulary)
model.set_state_dict(paddle.load(os.path.join(args.model_path, 'model.pdparams')))
model.eval()


# 评估模型
def evaluate():
    cer = []
    for batch_id, (inputs, labels, _, _) in enumerate(test_loader()):
        # 执行识别
        outs = model(inputs)
        outs = paddle.nn.functional.softmax(outs, 1)
        outs = paddle.transpose(outs, perm=[0, 2, 1])
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
