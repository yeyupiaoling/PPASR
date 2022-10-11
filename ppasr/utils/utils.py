import distutils.util
import json
import os
import time
import wave

import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from zhconv import convert
from tn.chinese.normalizer import Normalizer


def print_arguments(args, configs):
    print("----------- 额外配置参数 -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")
    print("----------- 配置文件参数 -----------")
    for arg, value in sorted(configs.items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


def labels_to_string(label, vocabulary, blank_index=0):
    labels = []
    for l in label:
        index_list = [index for index in l if index != blank_index and index != -1]
        labels.append((''.join([vocabulary[index] for index in index_list])).replace('<space>', ' '))
    return labels


# 使用模糊删除方式删除文件
def fuzzy_delete(dir, fuzzy_str):
    if os.path.exists(dir):
        for file in os.listdir(dir):
            if fuzzy_str in file:
                path = os.path.join(dir, file)
                os.remove(path)


# 创建数据列表
def create_manifest(annotation_path, train_manifest_path, test_manifest_path, is_change_frame_rate=True, max_test_manifest=10000):
    data_list = []
    test_list = []
    durations = []
    normalizer = Normalizer()
    for annotation_text in os.listdir(annotation_path):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        if os.path.splitext(annotation_text_path)[-1] == '.json':
            with open(annotation_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                d = json.loads(line)
                audio_path, text = d["audio_filepath"], d["text"]
                start_time, end_time, duration = d["start_time"], d["end_time"], d["duration"]
                # 重新调整音频格式并保存
                if is_change_frame_rate:
                    change_rate(audio_path)
                # 获取音频长度
                durations.append(duration)
                # 对文本进行标准化
                text = normalizer.normalize(text).lower()
                # 过滤非法的字符
                text = is_ustr(text)
                if len(text) == 0: continue
                # 保证全部都是简体
                text = convert(text, 'zh-cn')
                # 加入数据列表中
                line = dict(audio_filepath=audio_path.replace('\\', '/'),
                            text=text,
                            duration=duration,
                            start_time=start_time,
                            end_time=end_time)
                if annotation_text == 'test.json':
                    test_list.append(line)
                else:
                    data_list.append(line)
        else:
            with open(annotation_text_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                audio_path = line.split('\t')[0]
                # 重新调整音频格式并保存
                if is_change_frame_rate:
                    change_rate(audio_path)
                # 获取音频长度
                audio_data, samplerate = soundfile.read(audio_path)
                duration = float(len(audio_data)) / samplerate
                durations.append(duration)
                # 对文本进行标准化
                text = normalizer.normalize(line.split('\t')[1].replace('\n', '').replace('\r', '')).lower()
                # 过滤非法的字符
                text = is_ustr(text)
                if len(text) == 0:continue
                # 保证全部都是简体
                text = convert(text, 'zh-cn')
                # 加入数据列表中
                line = dict(audio_filepath=audio_path.replace('\\', '/'),
                            text=text,
                            duration=duration)
                if annotation_text == 'test.txt':
                    test_list.append(line)
                else:
                    data_list.append(line)

    # 按照音频长度降序
    data_list.sort(key=lambda x: x["duration"], reverse=False)
    if len(test_list) > 0:
        test_list.sort(key=lambda x: json.loads(x)["duration"], reverse=False)
    # 数据写入到文件中
    f_train = open(train_manifest_path, 'w', encoding='utf-8')
    f_test = open(test_manifest_path, 'w', encoding='utf-8')
    for line in test_list:
        f_test.write('{}\n'.format(str(line).replace("'", '"')))
    interval = 500
    if len(data_list) / 500 > max_test_manifest:
        interval = len(data_list) // max_test_manifest
    for i, line in enumerate(data_list):
        if i % interval == 0:
            if len(test_list) == 0:
                f_test.write('{}\n'.format(str(line).replace("'", '"')))
            else:
                f_train.write('{}\n'.format(str(line).replace("'", '"')))
        else:
            f_train.write('{}\n'.format(str(line).replace("'", '"')))
    f_train.close()
    f_test.close()
    print("完成生成数据列表，数据集总长度为{:.2f}小时！".format(sum(durations) / 3600.))


# 将多段短音频合并为长音频，减少文件数量
def merge_audio(annotation_path, save_audio_path, max_duration=600):
    # 合并数据列表
    train_list_path = os.path.join(annotation_path, 'merge_audio.json')
    if os.path.exists(train_list_path):
        f_ann = open(train_list_path, 'a', encoding='utf-8')
    else:
        f_ann = open(train_list_path, 'w', encoding='utf-8')
    wav, duration_sum, list_data = [], [], []
    for annotation_text in os.listdir(annotation_path):
        annotation_text_path = os.path.join(annotation_path, annotation_text)
        if os.path.splitext(annotation_text_path)[-1] != '.txt':continue
        if os.path.splitext(annotation_text_path)[-1] == 'test.txt':continue
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            audio_path, text = line.replace('\n', '').replace('\r', '').split('\t')
            if not os.path.exists(audio_path):continue
            audio_data, samplerate = soundfile.read(audio_path)
            # 获取音频长度
            duration = float(len(audio_data)) / samplerate
            # 重新调整音频格式并保存
            if samplerate != 16000:
                audio_data = librosa.resample(audio_data, samplerate, target_sr=16000)
                soundfile.write(audio_path, audio_data, samplerate=16000)
                audio_data, _ = soundfile.read(audio_path)
            # 合并数据
            duration_sum.append(duration)
            wav.append(audio_data)
            # 列表数据
            list_d = dict(text=text,
                          duration=round(duration, 5),
                          start_time=round(sum(duration_sum) - duration, 5),
                          end_time=round(sum(duration_sum), 5))
            list_data.append(list_d)
            # 删除已处理的音频文件
            # os.remove(audio_path)
            # 保存合并音频文件
            if sum(duration_sum) >= max_duration:
                # 保存路径
                dir_num = len(os.listdir(save_audio_path)) -1 if os.path.exists(save_audio_path) else 0
                save_dir = os.path.join(save_audio_path, str(dir_num))
                os.makedirs(save_dir, exist_ok=True)
                if len(os.listdir(save_dir)) >= 1000:
                    save_dir = os.path.join(save_audio_path, str(dir_num + 1))
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.wav').replace('\\', '/')
                data = np.concatenate(wav)
                soundfile.write(save_path, data=data, samplerate=16000, format='WAV')
                # 写入到列表文件
                for list_d in list_data:
                    list_d['audio_filepath'] = save_path
                    f_ann.write('{}\n'.format(str(list_d).replace("'", '"')))
                f_ann.flush()
                wav, duration_sum, list_data = [], [], []
        # 删除已处理的标注文件
        os.remove(annotation_text_path)
    f_ann.close()


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
    return out_str


# 判断是否为中文字符或者英文字符
def is_uchar(uchar):
    if uchar == ' ':return True
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    if uchar in ('-', ',', '.', '>', '?'):
        return False
    return False


# 生成噪声的数据列表
def create_noise(path, noise_manifest_path, is_change_frame_rate=True):
    if not os.path.exists(path):
        print('噪声音频文件为空，已跳过！')
        return
    json_lines = []
    print('正在创建噪声数据列表，路径：%s，请等待 ...' % path)
    for file in tqdm(os.listdir(path)):
        audio_path = os.path.join(path, file)
        try:
            # 噪声的标签可以标记为空
            text = ""
            # 重新调整音频格式并保存
            if is_change_frame_rate:
                change_rate(audio_path)
            f_wave = wave.open(audio_path, "rb")
            duration = f_wave.getnframes() / f_wave.getframerate()
            json_lines.append(
                json.dumps(
                    {
                        'audio_filepath': audio_path.replace('\\', '/'),
                        'duration': duration,
                        'text': text
                    },
                    ensure_ascii=False))
        except Exception as e:
            continue
    with open(noise_manifest_path, 'w', encoding='utf-8') as f_noise:
        for json_line in json_lines:
            f_noise.write(json_line + '\n')


# 获取全部字符
def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            for char in line["text"].replace('\n', ''):
                counter.update(char)
    if os.path.exists(manifest_path.replace('train', 'test')):
        with open(manifest_path.replace('train', 'test'), 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                line = json.loads(line)
                for char in line["text"].replace('\n', ''):
                    counter.update(char)
