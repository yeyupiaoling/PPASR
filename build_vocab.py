import argparse
import functools
import codecs
from collections import Counter
from data.utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('count_threshold',  int,    0,  "字符计数的截断阈值，0为不做限制")
add_arg('vocab_path',       str,    '../dataset/zh_vocab.json', "生成的数据字典文件")
add_arg('manifest_path',    str,    '../dataset/manifest.train', "manifest path")
args = parser.parse_args()


def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            for char in line.split(',')[1].replace('\n', ''):
                counter.update(char)


def main():
    print_arguments(args)

    counter = Counter()
    count_manifest(counter, args.manifest_path)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with codecs.open(args.vocab_path, 'w', 'utf-8') as fout:
        labels = ['?']
        for char, count in count_sorted:
            if count < args.count_threshold: break
            labels.append(char)
        fout.write(str(labels))
    print("完成！")


if __name__ == '__main__':
    main()
