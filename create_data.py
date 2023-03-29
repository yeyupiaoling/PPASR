import argparse
import functools

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',              str,  'configs/conformer.yml',    '配置文件')
add_arg('annotation_path',      str,  'dataset/annotation/',      '标注文件的路径')
add_arg('noise_path',           str,  'dataset/audio/noise',      '噪声音频存放的文件夹路径')
add_arg('save_audio_path',      str,  'dataset/audio/merge_audio','合并音频的保存路径')
add_arg('is_change_frame_rate', bool, False,        '是否统一改变音频的采样率')
add_arg('only_keep_zh_en',      bool, True,         '是否只保留中文和英文字符，训练其他语言可以设置为False')
add_arg('max_test_manifest',    int,  10000,        '生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据')
add_arg('count_threshold',      int,  2,            '字符计数的截断阈值，0为不做限制')
add_arg('num_workers',          int,  8,            '读取数据的线程数量')
add_arg('num_samples',          int,  1000000,      '用于计算均值和标准值得音频数量，当为-1使用全部数据')
add_arg('is_merge_audio',       bool, False,        '是否将多个短音频合并成长音频，以减少音频文件数量，注意会自动删除原始音频文件')
add_arg('max_duration',         int,  600,          '合并音频的最大长度，单位秒')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = PPASRTrainer(configs=args.configs)

# 创建训练数据列表和归一化文件
trainer.create_data(annotation_path=args.annotation_path,
                    noise_path=args.noise_path,
                    num_samples=args.num_samples,
                    count_threshold=args.count_threshold,
                    is_change_frame_rate=args.is_change_frame_rate,
                    max_test_manifest=args.max_test_manifest,
                    only_keep_zh_en=args.only_keep_zh_en,
                    is_merge_audio=args.is_merge_audio,
                    save_audio_path=args.save_audio_path,
                    max_duration=args.max_duration)
