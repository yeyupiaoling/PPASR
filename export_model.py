import argparse
import functools

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,   'deepspeech2',              '所使用的模型', choices=['deepspeech2', 'deepspeech2_big'])
add_arg('dataset_vocab',    str,   'dataset/vocabulary.txt',   '数据字典的路径')
add_arg('mean_std_path',    str,   'dataset/mean_std.npz',     '数据集的均值和标准值的npy文件路径')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('feature_method',   str,   'linear',                   '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
add_arg('resume_model',     str,   'models/deepspeech2/best_model/',  '准备转换的模型路径')
args = parser.parse_args()
print_arguments(args)


trainer = PPASRTrainer(use_model=args.use_model,
                       feature_method=args.feature_method,
                       mean_std_path=args.mean_std_path,
                       dataset_vocab=args.dataset_vocab)

trainer.export(save_model_path=args.save_model, resume_model=args.resume_model)
