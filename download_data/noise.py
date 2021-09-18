import argparse
import os
import functools
import shutil

from utility import download, unzip
from utility import add_arguments, print_arguments

DATA_URL = 'http://www.openslr.org/resources/28/rirs_noises.zip'
MD5_DATA = 'e6f48e257286e05de56413b4779d8ffb'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
parser.add_argument("--target_dir",
                    default="../dataset/audio/",
                    type=str,
                    help="存放音频文件的目录 (默认: %(default)s)")
parser.add_argument("--noise_path",
                    default="../dataset/audio/noise/",
                    type=str,
                    help="存放噪声音频的目录 (默认: %(default)s)")
args = parser.parse_args()


def prepare_dataset(url, md5sum, target_dir, noise_path):
    """Download, unpack and move noise file."""
    data_dir = os.path.join(target_dir, 'RIRS_NOISES')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unzip(filepath, target_dir)
        os.remove(filepath)
    else:
        print("Skip downloading and unpacking. RIRS_NOISES data already exists in %s." % target_dir)
    # 移动噪声音频到指定文件夹
    if not os.path.exists(noise_path):
        os.makedirs(noise_path)
    json_lines = []
    data_types = [
        'pointsource_noises', 'real_rirs_isotropic_noises', 'simulated_rirs'
    ]
    for dtype in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, dtype)
        for subfolder, _, filelist in sorted(os.walk(audio_dir)):
            for fname in filelist:
                if '.wav' not in fname:continue
                audio_path = os.path.join(subfolder, fname)
                shutil.move(audio_path, os.path.join(noise_path, fname))
    shutil.rmtree(data_dir, ignore_errors=True)


def main():
    print_arguments(args)
    if args.target_dir.startswith('~'):
        args.target_dir = os.path.expanduser(args.target_dir)

    prepare_dataset(url=DATA_URL,
                    md5sum=MD5_DATA,
                    target_dir=args.target_dir,
                    noise_path=args.noise_path)


if __name__ == '__main__':
    main()
