import argparse
import os
import functools
import shutil

from utility import download, unzip
from utility import add_arguments, print_arguments

DATA_URL = 'https://openslr.trmal.net/resources/28/rirs_noises.zip'
MD5_DATA = 'e6f48e257286e05de56413b4779d8ffb'

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("target_dir", default="../dataset/audio/", type=str, help="存放音频文件的目录")
add_arg("noise_path", default="../dataset/noise/", type=str, help="存放噪声文件的目录")
add_arg("reverb_path", default="../dataset/reverb/", type=str, help="存放混响文件的目录")
args = parser.parse_args()
print_arguments(args)


def prepare_dataset(url, md5sum, target_dir, noise_path, reverb_path):
    """Download, unpack and move noise file."""
    data_dir = os.path.join(target_dir, 'RIRS_NOISES')
    if not os.path.exists(data_dir):
        filepath = download(url, md5sum, target_dir)
        unzip(filepath, target_dir)
        os.remove(filepath)
    else:
        print("Skip downloading and unpacking. RIRS_NOISES data already exists in %s." % target_dir)
    # 移动噪声音频到指定文件夹
    os.makedirs(noise_path, exist_ok=True)
    os.makedirs(reverb_path, exist_ok=True)
    noise_list_path = os.path.join(data_dir, 'real_rirs_isotropic_noises/noise_list')
    reverb_list_path = os.path.join(data_dir, 'real_rirs_isotropic_noises/rir_list')
    copy_files(os.path.dirname(data_dir), noise_path, noise_list_path)
    copy_files(os.path.dirname(data_dir), reverb_path, reverb_list_path)
    shutil.rmtree(data_dir, ignore_errors=True)


def copy_files(src_dir, dst_dir, list_path):
    with open(list_path, 'r') as f:
        for line in f:
            path = line.strip().split(' ')[-1]
            filename = os.path.basename(path)
            src_file = os.path.join(src_dir, path)
            dst_file = os.path.join(dst_dir, filename)
            shutil.move(src_file, dst_file)


def main():
    prepare_dataset(url=DATA_URL,
                    md5sum=MD5_DATA,
                    target_dir=args.target_dir,
                    noise_path=args.noise_path,
                    reverb_path=args.reverb_path)


if __name__ == '__main__':
    main()
