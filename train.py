import argparse
import functools
import os

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,    'configs/conformer.yml',    '配置文件')
add_arg('data_augment_configs', str,    'configs/augmentation.yml', '数据增强配置文件')
add_arg("local_rank",           int,    0,                          '多卡训练的本地GPU')
add_arg("use_gpu",              bool,   True,                       '是否使用GPU训练')
add_arg('metrics_type',         str,    'cer',                      '评估指标类型，中文用cer，英文用wer')
add_arg('save_model_path',      str,    'models/',                  '模型保存的路径')
add_arg('resume_model',         str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model',     str,    None,                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()

if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    print_arguments(args=args)

# 获取训练器
trainer = PPASRTrainer(configs=args.configs,
                       use_gpu=args.use_gpu,
                       metrics_type=args.metrics_type,
                       data_augment_configs=args.data_augment_configs)

trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
