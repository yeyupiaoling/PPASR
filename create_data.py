import argparse
import functools

import yaml

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,  'configs/config_zh.yml',    '配置文件')
add_arg('annotation_path',      str,  'dataset/annotation/',      '标注文件的路径')
add_arg('noise_path',           str,  'dataset/audio/noise',      '噪声音频存放的文件夹路径')
add_arg('is_change_frame_rate', bool, True,         '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('max_test_manifest',    int,  10000,        '生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据')
add_arg('count_threshold',      int,  2,            '字符计数的截断阈值，0为不做限制')
add_arg('num_workers',          int,  8,            '读取数据的线程数量')
add_arg('num_samples',          int,  1000000,      '用于计算均值和标准值得音频数量，当为-1使用全部数据')
args = parser.parse_args()

# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)

# 获取训练器
trainer = PPASRTrainer(configs=configs)

# 创建训练数据列表和归一化文件
trainer.create_data(annotation_path=args.annotation_path,
                    noise_path=args.noise_path,
                    num_samples=args.num_samples,
                    count_threshold=args.count_threshold,
                    is_change_frame_rate=args.is_change_frame_rate,
                    max_test_manifest=args.max_test_manifest)
