import argparse
import functools
import json
import os
import wave
from collections import Counter

import librosa
import soundfile
from tqdm import tqdm

from utils.utils import add_arguments, print_arguments
from data_utils.normalizer import FeatureNormalizer

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('--annotation_path',    str,  'dataset/annotation/',      '标注文件的路径')
add_arg('manifest_prefix',      str,  'dataset/',                 '训练数据清单，包括音频路径和标注信息')
add_arg('is_change_frame_rate', bool, True,                       '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('count_threshold',      int,  0,                          '字符计数的截断阈值，0为不做限制')
add_arg('vocab_path',           str,  'dataset/vocabulary.json',  '生成的数据字典文件')
add_arg('manifest_path',        str,  'dataset/manifest.train',   '数据列表路径')
add_arg('num_workers',          int,   8,                         '读取数据的线程数量')
add_arg('num_samples',          int,  -1,                         '用于计算均值和标准值得音频数量，当为-1使用全部数据')
add_arg('output_path',          str,  './dataset/mean_std.npz',   '保存均值和标准值得numpy文件路径，后缀 (.npz).')
args = parser.parse_args()


# 创建数据列表
def create_manifest(annotation_path, manifest_path_prefix):
    data_list = []
    durations = []
    for annotation_text in os.listdir(annotation_path):
        annotation_text = os.path.join(annotation_path, annotation_text)
        with open(annotation_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path = line.split('\t')[0]
            # 重新调整音频格式并保存
            if args.is_change_frame_rate:
                change_rate(audio_path)
            # 获取音频长度
            f_wave = wave.open(audio_path, "rb")
            duration = f_wave.getnframes() / f_wave.getframerate()
            durations.append(duration)
            # 过滤非法的字符
            text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
            # 加入数据列表中
            line = '{"audio_path":"%s", "duration":%.2f, "text":"%s"}' % (audio_path.replace('\\', '/'), duration, text)
            data_list.append(line)

    # 按照音频长度降序
    data_list.sort(key=lambda x: json.loads(x)["duration"], reverse=True)
    # 数据写入到文件中
    f_train = open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', encoding='utf-8')
    f_test = open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', encoding='utf-8')
    for i, line in enumerate(data_list):
        if i % 100 == 0:
            f_test.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_test.close()
    print("完成生成数据列表，数据集总长度为{:.2f}小时！".format(sum(durations) / 3600.))


# 改变音频采样率为16000Hz
def change_rate(audio_path):
    data, sr = soundfile.read(audio_path)
    if sr != 16000:
        data = librosa.resample(data, sr, target_sr=16000)
        soundfile.write(audio_path, data, samplerate=16000)


# 过滤非法的字符
def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
        else:
            out_str = out_str + ' '
    return ''.join(out_str.split())


# 判断是否为中文文字字符
def is_uchar(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return False
    if uchar in ('-', ',', '.', '>', '?'):
        return False
    return False


# 获取全部字符
def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            for char in line["text"].replace('\n', ''):
                counter.update(char)


# 计算数据集的均值和标准值
def compute_mean_std(manifest_path, num_samples, output_path):
    normalizer = FeatureNormalizer(mean_std_filepath=None,
                                   manifest_path=manifest_path,
                                   num_samples=num_samples,
                                   num_workers=args.num_workers)
    # 将计算的结果保存的文件中
    normalizer.write_to_file(output_path)
    print('计算的均值和标准值已保存在 %s！' % output_path)


def main():
    print_arguments(args)
    print('开始生成数据列表...')
    create_manifest(annotation_path=args.annotation_path,
                    manifest_path_prefix=args.manifest_prefix)

    print('开始生成数据字典...')
    counter = Counter()
    count_manifest(counter, args.manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(args.vocab_path, 'w', encoding='utf-8') as fout:
        labels = ['?']
        for char, count in count_sorted:
            if count < args.count_threshold: break
            labels.append(char)
        fout.write(str(labels).replace("'", '"'))
    print('数据字典生成完成！')

    print('开始抽取%s条数据计算均值和标准值...' % args.num_samples)
    compute_mean_std(args.manifest_path, args.num_samples, args.output_path)


if __name__ == '__main__':
    main()
