import argparse
import functools
import os
import wave

from collections import Counter
from data.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('--annotation_path',    str,  '../dataset/annotation/',   '标注文件的路径')
add_arg('manifest_prefix',      str,  '../dataset/',               '训练数据清单，包括音频路径和标注信息')
add_arg('is_change_frame_rate', bool, False,                       '是否统一改变音频为16000Hz，这会消耗大量的时间')
add_arg('count_threshold',      int,  0,                           '字符计数的截断阈值，0为不做限制')
add_arg('vocab_path',           str,  '../dataset/zh_vocab.json',  '生成的数据字典文件')
add_arg('manifest_path',        str,  '../dataset/manifest.train', 'manifest path')
args = parser.parse_args()


def create_manifest(annotation_path, manifest_path_prefix):
    data_list = []
    for annotation_text in os.listdir(annotation_path):
        annotation_text = os.path.join(annotation_path, annotation_text)
        with open(annotation_text, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
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

            text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
            data_list.append(audio_path + ',' + text)

    f_train = open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', encoding='utf-8')
    f_dev = open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', encoding='utf-8')
    for i, line in enumerate(data_list):
        if i % 100 == 0:
            f_dev.write(line + '\n')
        else:
            f_train.write(line + '\n')
    f_train.close()
    f_dev.close()
    print('done.')


def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
        else:
            out_str = out_str + ' '
    return ''.join(out_str.split())


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


def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for char in line.split(',')[1].replace('\n', ''):
                counter.update(char)


def main():
    print_arguments(args)
    create_manifest(annotation_path=args.annotation_path,
                    manifest_path_prefix=args.manifest_prefix)

    counter = Counter()
    count_manifest(counter, args.manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(args.vocab_path, 'w', encoding='utf-8') as fout:
        labels = ['?']
        for char, count in count_sorted:
            if count < args.count_threshold: break
            labels.append(char)
        fout.write(str(labels).replace("'", '"'))
    print('完成！')


if __name__ == '__main__':
    main()
