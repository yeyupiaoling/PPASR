import argparse
import codecs
import functools
import os
import wave

from data.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--annotation_path",
                    default="../dataset/annotation/",
                    type=str,
                    help="标注文件的路径。 (default: %(default)s)")
parser.add_argument("--manifest_prefix",
                    default="../dataset/",
                    type=str,
                    help="训练数据清单，包括音频路径和标注信息。 (default: %(default)s)")
parser.add_argument("--is_change_frame_rate",
                    default=False,
                    type=bool,
                    help="是否统一改变音频为16000Hz，这会消耗大量的时间。 (default: %(default)s)")
args = parser.parse_args()


def create_manifest(annotation_path, manifest_path_prefix):
    data_list = []
    for annotation_text in os.listdir(annotation_path):
        annotation_text = os.path.join(annotation_path, annotation_text)
        with codecs.open(annotation_text, 'r', 'utf-8') as f:
            lines = f.readlines()
        for line in lines:
            audio_path = line.split('\t')[0]
            # 重新调整音频格式并保存
            if args.is_change_frame_rate:
                f = wave.open(audio_path, "rb")
                str_data = f.readframes(f.getnframes())
                f.close()
                file = wave.open(audio_path, 'wb')
                file.setnchannels(1)
                file.setsampwidth(4)
                file.setframerate(16000)
                file.writeframes(str_data)
                file.close()

            text = is_ustr(line.split('\t')[1].replace('\n', '').replace('\r', ''))
            data_list.append(audio_path + "," + text)

    f_train = codecs.open(os.path.join(manifest_path_prefix, 'manifest.train'), 'w', 'utf-8')
    f_dev = codecs.open(os.path.join(manifest_path_prefix, 'manifest.test'), 'w', 'utf-8')
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


def main():
    print_arguments(args)
    create_manifest(annotation_path=args.annotation_path,
                    manifest_path_prefix=args.manifest_prefix)


if __name__ == '__main__':
    main()
