import argparse
import functools
import os
import time
import numpy as np
import paddle

from data.utility import add_arguments, print_arguments
from data_utils.audio_featurizer import AudioFeaturizer
from data_utils.normalizer import FeatureNormalizer
from utils.decoder import GreedyDecoder
from model_utils.deepspeech2 import DeepSpeech2Model


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_conv_layers',  int,   2,                        '卷积层数量')
add_arg('num_rnn_layers',   int,   3,                        '循环神经网络的数量')
add_arg('rnn_layer_size',   int,   1024,                     '循环神经网络的大小')
add_arg('audio_path',       str,  'dataset/test.wav',        '用于识别的音频路径')
add_arg('dataset_vocab',    str,  'dataset/vocabulary.json', '数据字典的路径')
add_arg('model_path',       str,  'models/step_final/',      '模型的路径')
add_arg('mean_std_path',    str,  'dataset/mean_std.npz',    '数据集的均值和标准值的npy文件路径')
args = parser.parse_args()


print_arguments(args)
# 加载数据字典
with open(args.dataset_vocab, 'r', encoding='utf-8') as f:
    labels = eval(f.read())
vocabulary = dict([(labels[i], i) for i in range(len(labels))])

# 获取解码器
greedy_decoder = GreedyDecoder(vocabulary)
# 提取音频特征器和归一化器
audio_featurizer = AudioFeaturizer()
normalizer = FeatureNormalizer(mean_std_filepath=args.mean_std_path)

# 创建模型
model = DeepSpeech2Model(feat_size=audio_featurizer.feature_dim(),
                         dict_size=len(vocabulary),
                         num_conv_layers=args.num_conv_layers,
                         num_rnn_layers=args.num_rnn_layers,
                         rnn_size=args.rnn_layer_size)
model.set_state_dict(paddle.load(os.path.join(args.model_path, 'model.pdparams')))
model.eval()


@paddle.no_grad()
def infer():
    # 提取音频特征
    audio = audio_featurizer.load_audio_file(args.audio_path)
    feature = audio_featurizer.featurize(audio)
    # 对特征归一化
    audio = normalizer.apply(feature)[np.newaxis, :]
    audio = paddle.to_tensor(audio, dtype=paddle.float32)
    audio_len = paddle.to_tensor(feature.shape[1], dtype=paddle.int64)
    # 执行识别
    out, _ = model(audio, audio_len)
    out = paddle.nn.functional.softmax(out, 2)
    # 执行解码
    out_string, out_offset = greedy_decoder.decode(out)
    return out_string


if __name__ == '__main__':
    start = time.time()
    result_text = infer()[0][0]
    end = time.time()
    print('识别时间：%dms，识别结果：%s' % (round((end - start) * 1000), result_text))
