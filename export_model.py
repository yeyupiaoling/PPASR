import argparse
import functools

import yaml

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/config_zh.yml',    '配置文件')
add_arg('save_model',       str,   'models/',                  '模型保存的路径')
add_arg('resume_model',     str,   'models/{}_{}/best_model/', '准备转换的模型路径')
args = parser.parse_args()


# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)

# 获取训练器
trainer = PPASRTrainer(configs=configs)

# 导出预测模型
trainer.export(save_model_path=args.save_model,
               resume_model=args.resume_model.format(configs['use_model'], configs['preprocess']['feature_method']))
