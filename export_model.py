import argparse
import functools
import os
from datetime import datetime

import paddle
import paddle.distributed as dist
from paddle.static import InputSpec

from data_utils.audio_featurizer import AudioFeaturizer
from utils.utils import add_arguments, print_arguments
from model_utils.deepspeech2 import DeepSpeech2Model

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('num_conv_layers',  int,   2,                          '卷积层数量')
add_arg('num_rnn_layers',   int,   3,                          '循环神经网络的数量')
add_arg('rnn_layer_size',   int,   1024,                       '循环神经网络的大小')
add_arg('dataset_vocab',    str,   'dataset/vocabulary.json',  '数据字典的路径')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('resume',           str,   'models/epoch_50',          '恢复训练，当为None则不使用预训练模型')
args = parser.parse_args()


def export(args):
    # 获取训练数据
    audio_featurizer = AudioFeaturizer()
    with open(args.dataset_vocab, 'r', encoding='utf-8') as f:
        vocabulary = eval(f.read())
    # 获取模型
    model = DeepSpeech2Model(feat_size=audio_featurizer.feature_dim,
                             dict_size=len(vocabulary),
                             num_conv_layers=args.num_conv_layers,
                             num_rnn_layers=args.num_rnn_layers,
                             rnn_size=args.rnn_layer_size)
    if dist.get_rank() == 0:
        print('[{}] input_size的第三个参数是变长的，这里为了能查看输出的大小变化，指定了一个值！'.format(datetime.now()))
        paddle.summary(model, input_size=[(None, audio_featurizer.feature_dim, 970), (None,)], dtypes=[paddle.float32, paddle.int64])

    # 加载预训练模型
    resume_model_path = os.path.join(args.resume, 'model.pdparams')
    assert os.path.join(resume_model_path), "恢复模型不存在！"
    model.set_state_dict(paddle.load(resume_model_path))
    print('[{}] 成功恢复模型参数和优化方法参数'.format(datetime.now()))

    infer_model_path = os.path.join(args.save_model, 'infer')
    if not os.path.exists(infer_model_path):
        os.makedirs(infer_model_path)
    paddle.jit.save(layer=model,
                    path=os.path.join(infer_model_path, 'model'),
                    input_spec=[InputSpec(shape=(-1, audio_featurizer.feature_dim, -1), dtype=paddle.float32),
                                InputSpec(shape=(-1,), dtype=paddle.int64)])
    print("预测模型已保存：%s" % infer_model_path)


if __name__ == '__main__':
    print_arguments(args)
    export(args)
