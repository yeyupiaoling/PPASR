import argparse
import functools
import os
import time

import paddle

from data.utility import add_arguments, print_arguments
from utils.data import load_audio_mfcc
from utils.decoder import GreedyDecoder
from utils.model import PPASR


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('audio_path',    str,  'dataset/test.wav',       '用于识别的音频路径')
add_arg('dataset_vocab', str,  'dataset/zh_vocab.json',  '数据字典的路径')
add_arg('model_path',    str,  'models/step_final/',     '模型的路径')
args = parser.parse_args()


print_arguments(args)
# 加载数据字典
with open(args.dataset_vocab, 'r', encoding='utf-8') as f:
    labels = eval(f.read())
vocabulary = dict([(labels[i], i) for i in range(len(labels))])
# 获取解码器
greedy_decoder = GreedyDecoder(vocabulary)

# 创建模型
model = PPASR(vocabulary)
model.set_state_dict(paddle.load(os.path.join(args.model_path, 'model.pdparams')))
# 获取保存在模型中的数据均值和标准值
data_mean = model.data_mean.numpy()[0]
data_std = model.data_std.numpy()[0]
model.eval()


def infer():
    # 读取音频文件转成梅尔频率倒谱系数(MFCCs)
    mfccs = load_audio_mfcc(args.audio_path, mean=data_mean, std=data_std)

    mfccs = paddle.to_tensor(mfccs, dtype='float32')
    mfccs = paddle.unsqueeze(mfccs, axis=0)
    # 执行识别
    out = model(mfccs)
    out = paddle.nn.functional.softmax(out, 1)
    out = paddle.transpose(out, perm=[0, 2, 1])
    # 执行解码
    out_string, out_offset = greedy_decoder.decode(out)
    return out_string


if __name__ == '__main__':
    start = time.time()
    result_text = infer()
    end = time.time()
    print('识别时间：%dms，识别结果：%s' % (round((end - start) * 1000), result_text))
