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

    if os.path.exists(os.path.join(annotation_dir, 'wenetspeech.txt')):
        f_ann = open(os.path.join(annotation_dir, 'wenetspeech.txt'), 'a', encoding='utf-8')
    else:
        f_ann = open(os.path.join(annotation_dir, 'wenetspeech.txt'), 'w', encoding='utf-8')

    if os.path.exists(os.path.join(annotation_dir, 'test.txt')):
        f_ann_test = open(os.path.join(annotation_dir, 'test.txt'), 'a', encoding='utf-8')
    else:
        f_ann_test = open(os.path.join(annotation_dir, 'test.txt'), 'w', encoding='utf-8')

    with open(wenetspeech_json, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'audios.item')
        while True:
            try:
                long_audio = objects.__next__()
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
                    print(f'正在处理{long_audio_path}音频')
                    save_dir = long_audio_path[:-5]
                    os.makedirs(save_dir, exist_ok=True)
                    source_wav = AudioSegment.from_file(long_audio_path)
                    for segment_file in segments_lists:
                        try:
                            sid = segment_file['sid']
                            start_time = segment_file['begin_time']
                            end_time = segment_file['end_time']
                            text = segment_file['text']
                            confidence = segment_file['confidence']
                            if confidence < 0.95: continue
                        except Exception:
                            print(f'''Warning: {segment_file} something is wrong, skipped''')
                            continue
                        else:
                            start = int(start_time * 1000)
                            end = int(end_time * 1000)
                            target_audio = source_wav[start:end].set_frame_rate(16000)
                            save_audio_path = os.path.join(save_dir, sid.split('_')[-1] + '.wav')
                            target_audio.export(save_audio_path, format="wav")
                            if long_audio['path'].split('/')[1] != 'train':
                                f_ann_test.write('%s\t%s\n' % (save_audio_path, text))
                            else:
                                f_ann.write('%s\t%s\n' % (save_audio_path, text))
                    # 删除已经处理过的音频
                    os.remove(long_audio_path)
            except StopIteration:
                print("数据读取完成")
                break
        shutil.copy(os.path.join(annotation_dir, 'wenetspeech.txt'), os.path.join(input_dir, 'wenetspeech.txt'))
        shutil.copy(os.path.join(annotation_dir, 'test.txt'), os.path.join(input_dir, 'test.txt'))


if __name__ == '__main__':
    process_wenetspeech(wenetspeech_json=args.wenetspeech_json, annotation_dir=args.annotation_dir)