import argparse
import io
import os

from utility import download, unpack

URL_ROOT = "https://openslr.magicdatatech.com/resources/12"
URL_TEST_CLEAN = URL_ROOT + "/test-clean.tar.gz"
URL_TEST_OTHER = URL_ROOT + "/test-other.tar.gz"
URL_DEV_CLEAN = URL_ROOT + "/dev-clean.tar.gz"
URL_DEV_OTHER = URL_ROOT + "/dev-other.tar.gz"
URL_TRAIN_CLEAN_100 = URL_ROOT + "/train-clean-100.tar.gz"
URL_TRAIN_CLEAN_360 = URL_ROOT + "/train-clean-360.tar.gz"
URL_TRAIN_OTHER_500 = URL_ROOT + "/train-other-500.tar.gz"

MD5_TEST_CLEAN = "32fa31d27d2e1cad72775fee3f4849a9"
MD5_TEST_OTHER = "fb5a50374b501bb3bac4815ee91d3135"
MD5_DEV_CLEAN = "42e2234ba48799c1f50f24a7926300a1"
MD5_DEV_OTHER = "c8d0bcc9cca99d4f8b62fcc847357931"
MD5_TRAIN_CLEAN_100 = "2a93770f6d5c6c964bc36631d331a522"
MD5_TRAIN_CLEAN_360 = "c0e676e450a7ff2f54aeade5171606fa"
MD5_TRAIN_OTHER_500 = "d1a0fd59409feb2c614ce4d30c387708"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target_dir",
                    default="../dataset/audio/",
                    type=str,
                    help="存放音频文件的目录 (默认: %(default)s)")
parser.add_argument("--annotation_text",
                    default="../dataset/annotation/",
                    type=str,
                    help="存放音频标注文件的目录 (默认: %(default)s)")
args = parser.parse_args()


def create_annotation_text(data_dir, annotation_path):
    print('Create Librispeech annotation text ...')
    if not os.path.exists(annotation_path):
        os.makedirs(annotation_path)
    if not os.path.exists(os.path.join(annotation_path, 'test.txt')):
        f_train = open(os.path.join(annotation_path, 'librispeech.txt'), 'w', encoding='utf-8')
    else:
        f_train = open(os.path.join(annotation_path, 'librispeech.txt'), 'a', encoding='utf-8')
    if not os.path.exists(os.path.join(annotation_path, 'test.txt')):
        f_test = open(os.path.join(annotation_path, 'test.txt'), 'w', encoding='utf-8')
    else:
        f_test = open(os.path.join(annotation_path, 'test.txt'), 'a', encoding='utf-8')

    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        text_filelist = [filename for filename in filelist if filename.endswith('trans.txt')]
        if len(text_filelist) > 0:
            text_filepath = os.path.join(subfolder, text_filelist[0])
            for line in io.open(text_filepath, encoding="utf8"):
                segments = line.strip().split()
                text = ' '.join(segments[1:]).lower()
                audio_filepath = os.path.join(subfolder, segments[0] + '.flac')
                if 'test-clean' not in subfolder and 'test-other' not in subfolder and \
                        'dev-other' not in subfolder and 'dev-other' not in subfolder:
                    f_train.write(audio_filepath[3:] + '\t' + text + '\n')
                else:
                    if 'test-clean' in subfolder:
                        f_test.write(audio_filepath[3:] + '\t' + text + '\n')
    f_test.close()
    f_train.close()


def prepare_dataset(url, md5sum, target_dir, annotation_path):
    """Download, unpack and create summmary manifest file."""
    data_dir = os.path.join(target_dir, 'LibriSpeech')
    # download
    filepath = download(url, md5sum, target_dir)
    # unpack
    unpack(filepath, target_dir)

    create_annotation_text(data_dir, annotation_path)


def main():
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(url=URL_TEST_CLEAN,
                    md5sum=MD5_TEST_CLEAN,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)
    prepare_dataset(url=URL_DEV_CLEAN,
                    md5sum=MD5_DEV_CLEAN,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)
    prepare_dataset(url=URL_TRAIN_CLEAN_100,
                    md5sum=MD5_TRAIN_CLEAN_100,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)
    prepare_dataset(url=URL_TEST_OTHER,
                    md5sum=MD5_TEST_OTHER,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)
    prepare_dataset(url=URL_DEV_OTHER,
                    md5sum=MD5_DEV_OTHER,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)
    prepare_dataset(url=URL_TRAIN_CLEAN_360,
                    md5sum=MD5_TRAIN_CLEAN_360,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)
    prepare_dataset(url=URL_TRAIN_OTHER_500,
                    md5sum=MD5_TRAIN_OTHER_500,
                    target_dir=args.target_dir,
                    annotation_path=args.annotation_text)


if __name__ == '__main__':
    main()
