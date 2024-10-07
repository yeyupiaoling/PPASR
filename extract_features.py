import argparse
import functools

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/conformer.yml',       '配置文件')
add_arg('save_dir',         str,     'dataset/features',        '保存特征的路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = PPASRTrainer(configs=args.configs)

# 提取特征保存文件
trainer.extract_features(save_dir=args.save_dir)
