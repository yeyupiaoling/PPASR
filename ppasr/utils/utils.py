import distutils.util
import os
import re
import urllib.request
import zipfile

from loguru import logger
from tqdm import tqdm


def print_arguments(args=None, configs=None, title=None):
    if args:
        logger.info("----------- 额外配置参数 -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info(f"{arg}: {value}")
        logger.info("------------------------------------------------")
    if configs:
        title = title if title else "配置文件参数"
        logger.info(f"----------- {title} -----------")
        for arg, value in sorted(configs.items()):
            if arg == 'vocabulary': value = str(value)[:30] + ' ......'
            if isinstance(value, dict):
                logger.info(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            logger.info(f"\t\t{a1}: {v1}")
                    else:
                        logger.info(f"\t{a}: {v}")
            else:
                logger.info(f"{arg}: {value}")
        logger.info("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


class DictObject(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = DictObject()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


# 解压ZIP文件
def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        logger.error('This is not zip')


def download(url: str, download_target: str):
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))


# 下载模型文件
def download_model(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)
    unzip_path = download_target[:-4]

    if os.path.exists(unzip_path) and not os.path.isdir(unzip_path):
        raise RuntimeError(f"{unzip_path} exists and is not a regular dir")

    if os.path.isdir(unzip_path):
        return unzip_path

    download(url=url, download_target=download_target)
    unzip_file(download_target, os.path.dirname(download_target))
    os.remove(download_target)
    return unzip_path


# 判断是否是英文单词
def is_english_word(word):
    # 正则表达式匹配英文单词，允许撇号和连字符
    pattern = re.compile(r"^[A-Za-z'-]+$")
    return pattern.match(word) is not None
