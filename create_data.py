import argparse
import functools

from ppasr.trainer import PPASRTrainer
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('annotation_path',      str,  'dataset/annotation/',      '标注文件的路径')
add_arg('train_manifest',       str,  'dataset/manifest.train',   '训练数据的数据列表路径')
add_arg('test_manifest',        str,  'dataset/manifest.test',    '测试数据的数据列表路径')
add_arg('is_change_frame_rate', bool, True,                       '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('max_test_manifest',    int,  10000,                      '生成测试数据列表的最大数量，如果annotation_path包含了test.txt，就全部使用test.txt的数据')
add_arg('count_threshold',      int,  2,                          '字符计数的截断阈值，0为不做限制')
add_arg('dataset_vocab',        str,  'dataset/vocabulary.txt',   '生成的数据字典文件')
add_arg('num_workers',          int,  8,                          '读取数据的线程数量')
add_arg('num_samples',          int,  1000000,                    '用于计算均值和标准值得音频数量，当为-1使用全部数据')
add_arg('mean_std_path',        str,  'dataset/mean_std.npz',     '保存均值和标准值得numpy文件路径，后缀 (.npz).')
add_arg('noise_path',           str,  'dataset/audio/noise',      '噪声音频存放的文件夹路径')
add_arg('noise_manifest_path',  str,  'dataset/manifest.noise',   '噪声数据列表的路径')
add_arg('feature_method',       str,  'linear',                   '音频预处理方法', choices=['linear', 'mfcc', 'fbank'])
args = parser.parse_args()
print_arguments(args)


trainer = PPASRTrainer(mean_std_path=args.mean_std_path,
                       feature_method=args.feature_method,
                       train_manifest=args.train_manifest,
                       test_manifest=args.test_manifest,
                       dataset_vocab=args.dataset_vocab,
                       num_workers=args.num_workers)

trainer.create_data(annotation_path=args.annotation_path,
                    noise_manifest_path=args.noise_manifest_path,
                    noise_path=args.noise_path,
                    num_samples=args.num_samples,
                    count_threshold=args.count_threshold,
                    is_change_frame_rate=args.is_change_frame_rate,
                    max_test_manifest=args.max_test_manifest)
