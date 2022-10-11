import argparse
import os
import shutil

import ijson
from pydub import AudioSegment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wenetspeech_json',  type=str,    default='/media/wenetspeech/WenetSpeech.json',  help="WenetSpeech的标注json文件路径")
parser.add_argument('--annotation_dir',    type=str,    default='../dataset/annotation/',    help="存放数量列表的文件夹路径")
args = parser.parse_args()


def process_wenetspeech(wenetspeech_json, annotation_dir):
    input_dir = os.path.dirname(wenetspeech_json)
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    # 训练数据列表
    train_list_path = os.path.join(annotation_dir, 'wenetspeech.json')
    if os.path.exists(train_list_path):
        f_ann = open(train_list_path, 'a', encoding='utf-8')
    else:
        f_ann = open(train_list_path, 'w', encoding='utf-8')
    # 测试数据列表
    test_list_path = os.path.join(annotation_dir, 'test.json')
    if os.path.exists(test_list_path):
        f_ann_test = open(test_list_path, 'a', encoding='utf-8')
    else:
        f_ann_test = open(test_list_path, 'w', encoding='utf-8')
    i = 0
    # 开始读取数据，因为文件太大，无法获取进度
    with open(wenetspeech_json, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'audios.item')
        while True:
            try:
                long_audio = objects.__next__()
                i += 1
                try:
                    long_audio_path = os.path.realpath(os.path.join(input_dir, long_audio['path']))
                    aid = long_audio['aid']
                    segments_lists = long_audio['segments']
                    assert (os.path.exists(long_audio_path))
                except AssertionError:
                    print(f'''Warning: {long_audio_path} 不存在或者已经处理过自动删除了，跳过''')
                    continue
                except Exception:
                    print(f'''Warning: {aid} 数据读取错误，跳过''')
                    continue
                else:
                    print(f'正在处理第{i}音频：{long_audio_path}')
                    source_wav = AudioSegment.from_file(long_audio_path)
                    target_audio = source_wav.set_frame_rate(16000)
                    save_audio_path = long_audio_path.replace('.opus', '.wav')
                    target_audio.export(save_audio_path, format="wav")
                    for segment_file in segments_lists:
                        try:
                            start_time = float(segment_file['begin_time'])
                            end_time = float(segment_file['end_time'])
                            text = segment_file['text']
                            confidence = segment_file['confidence']
                            if confidence < 0.95: continue
                        except Exception:
                            print(f'''Warning: {segment_file} something is wrong, skipped''')
                            continue
                        else:
                            line = dict(audio_filepath=save_audio_path,
                                        text=text,
                                        duration=round(end_time - start_time, 3),
                                        start_time=round(start_time, 3),
                                        end_time=round(end_time, 3))
                            if long_audio['path'].split('/')[1] != 'train':
                                f_ann_test.write('{}\n'.format(str(line).replace("'", '"')))
                            else:
                                f_ann.write('{}\n'.format(str(line).replace("'", '"')))
                            f_ann.flush()
                            f_ann_test.flush()
                    # 删除已经处理过的音频
                    os.remove(long_audio_path)
            except StopIteration:
                print("数据读取完成")
                break
        f_ann.close()
        f_ann_test.close()
        shutil.copy(train_list_path, os.path.join(input_dir, 'wenetspeech_train.json'))
        shutil.copy(test_list_path, os.path.join(input_dir, 'wenetspeech_test.json'))


if __name__ == '__main__':
    process_wenetspeech(wenetspeech_json=args.wenetspeech_json, annotation_dir=args.annotation_dir)