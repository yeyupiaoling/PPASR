import argparse
import functools
import os
import random
import wave

import librosa
import numpy as np
from tqdm import tqdm
from collections import Counter

import soundfile

from data.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('--annotation_path',    str,  'dataset/annotation/',   '标注文件的路径')
add_arg('manifest_prefix',      str,  'dataset/',              '训练数据清单，包括音频路径和标注信息')
add_arg('is_change_frame_rate', bool, False,                   '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('min_duration',         int,  0,                       '过滤最短的音频长度')
add_arg('max_duration',         int,  20,                      '过滤最长的音频长度，当为-1的时候不限制长度')
add_arg('count_threshold',      int,  0,                       '字符计数的截断阈值，0为不做限制')
add_arg('vocab_path',           str,  'dataset/zh_vocab.json',  '生成的数据字典文件')
add_arg('manifest_path',        str,  'dataset/manifest.train', 'manifest path')
args = parser.parse_args()


# 创建数据列表
def create_manifest(annotation_path, manifest_path_prefix):
    data_list = []
    duration_sum = 0
    for annotation_text in os.listdir(annotation_path):
        annotation_text = os.path.join(annotation_path, annotation_text)
        with open(annotation_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path = line.split('\t')[0]
            # 重新调整音频格式并保存
            if args.is_change_frame_rate:
                f = wave.open(audio_path, 'rb')
                str_data = f.readframes(f.getnframes())
                f.close()
                file = wave.open(audio_path, 'wb')
                file.setnchannels(1)
                file.setsampwidth(4)
                file.setframerate(16000)
                file.writeframes(str_data)
                file.close()
            # 获取音频长度
            audio_data, samplerate = soundfile.read(audio_path)
            duration = float(len(audio_data) / samplerate)
            if duration < args.min_duration:
                continue
            if args.max_duration != -1 and duration > args.max_duration:
                continue
            duration_sum += duration
            # 过滤非法的字符
            text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
            # 加入数据列表中
            data_list.append(audio_path + ',' + str(duration) + ',' + text)

    # 按照音频长度降序
    data_list.sort(key=lambda x: float(x.strip().split(",")[1]), reverse=True)
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
    print("完成生成数据列表，数据集总长度为{:.2f}小时！".format(duration_sum / 3600.))


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


# 计算数据集的均值和标准值
def compute_mean_std(manifest_path):
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for i, line in enumerate(tqdm(lines)):
            if i % 10 == 0:
                wav_path = line.split(',')[0]
                with wave.open(wav_path) as wav:
                    wav = np.frombuffer(wav.readframes(wav.getnframes()), dtype="int16").astype("float32")
                mfccs = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=128, n_fft=512, hop_length=128).astype("float32")
                spec, phase = librosa.magphase(mfccs)
                spec = np.log1p(spec)
                data.append(spec)
    data = np.array(spec, dtype='float32')
    return data.mean(), data.std()


# 获取全部字符
def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            for char in line.split(',')[2].replace('\n', ''):
                counter.update(char)


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

    print('开始抽取10%的数据计算均值和标准值...')
    mean, std = compute_mean_std(args.manifest_path)
    print('【特别重要】：均值：%f, 标准值：%f, 请根据这两个值修改训练参数！' % (mean, std))


if __name__ == '__main__':
    main()