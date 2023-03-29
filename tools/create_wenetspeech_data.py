import argparse
import os
import shutil
import threading

import ijson
from pydub import AudioSegment
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--wenetspeech_json',  type=str,    default='F:\音频数据\WenetSpeech数据集/WenetSpeech.json',  help="WenetSpeech的标注json文件路径")
parser.add_argument('--annotation_dir',    type=str,    default='../dataset/annotation/',    help="存放数量列表的文件夹路径")
parser.add_argument('--to_wav',            type=bool,   default=False,                       help="是否把opus格式转换为wav格式，以空间换时间")
parser.add_argument('--num_workers',       type=int,    default=8,                           help="把opus格式转换为wav格式的线程数量")
args = parser.parse_args()


if not os.path.exists(args.annotation_dir):
    os.makedirs(args.annotation_dir)
# 训练数据列表
train_list_path = os.path.join(args.annotation_dir, 'wenetspeech.json')
f_ann = open(train_list_path, 'a', encoding='utf-8')
# 测试数据列表
test_list_path = os.path.join(args.annotation_dir, 'test.json')
f_ann_test = open(test_list_path, 'a', encoding='utf-8')

# 资源锁
threadLock = threading.Lock()
threads = []


class myThread(threading.Thread):
    def __init__(self, threadID, data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.data = data

    def run(self):
        print(f"开启线程：{self.threadID}，数据大小为：{len(self.data)}" )
        for i, data in enumerate(self.data):
            long_audio_path, segments_lists = data
            print(f'线程：{self.threadID} 正在处理：[{i+1}/{len(self.data)}]')
            lines = process_wenetspeech(long_audio_path, segments_lists)
            # 获取锁
            threadLock.acquire()
            for line in lines:
                if long_audio_path.split('/')[-4] != 'train':
                    f_ann_test.write('{}\n'.format(str(line).replace("'", '"')))
                else:
                    f_ann.write('{}\n'.format(str(line).replace("'", '"')))
                f_ann.flush()
                f_ann_test.flush()
            # 释放锁
            threadLock.release()
        print(f"线程：{self.threadID} 已完成")


# 处理WenetSpeech数据
def process_wenetspeech(long_audio_path, segments_lists):
    save_audio_path = long_audio_path.replace('.opus', '.wav')
    source_wav = AudioSegment.from_file(long_audio_path)
    target_audio = source_wav.set_frame_rate(16000)
    target_audio.export(save_audio_path, format="wav")
    lines = []
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
            lines.append(line)
    # 删除已经处理过的音频
    os.remove(long_audio_path)
    return lines


# 获取标注信息
def get_data(wenetspeech_json):
    data_list = []
    input_dir = os.path.dirname(wenetspeech_json)
    i = 0
    # 开始读取数据，因为文件太大，无法获取进度
    with open(wenetspeech_json, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'audios.item')
        print("开始读取数据")
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
                    data_list.append([long_audio_path.replace('\\', '/'), segments_lists])
            except StopIteration:
                print("数据读取完成")
                break
    return data_list


def main():
    all_data = get_data(args.wenetspeech_json)
    print(f'总数据量为：{len(all_data)}')
    if args.to_wav:
        text = input(f'音频文件将会转换为wav格式，这个过程可能很长，而且最终文件大小接近5.5T，是否继续？(y/n)')
        if text is None or text != 'y':
            return
        chunk_len = len(all_data) // args.num_workers
        for i in range(args.num_workers):
            sub_data = all_data[i * chunk_len: (i + 1) * chunk_len]
            thread = myThread(i, sub_data)
            thread.start()
            threads.append(thread)

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 复制标注文件，因为有些标注文件已经转换为wav文件
        input_dir = os.path.dirname(args.wenetspeech_json)
        shutil.copy(train_list_path, os.path.join(input_dir, 'wenetspeech_train.json'))
        shutil.copy(test_list_path, os.path.join(input_dir, 'wenetspeech_test.json'))
    else:
        text = input(f'将直接使用opus，值得注意的是opus读取速度会比wav格式慢很多，是否继续？(y/n)')
        if text is None or text != 'y':
            return
        for data in tqdm(all_data):
            long_audio_path, segments_lists = data
            for segment_file in segments_lists:
                start_time = float(segment_file['begin_time'])
                end_time = float(segment_file['end_time'])
                text = segment_file['text']
                confidence = segment_file['confidence']
                if confidence < 0.95: continue
                line = dict(audio_filepath=long_audio_path,
                            text=text,
                            duration=round(end_time - start_time, 3),
                            start_time=round(start_time, 3),
                            end_time=round(end_time, 3))
                if long_audio_path.split('/')[-4] != 'train':
                    f_ann_test.write('{}\n'.format(str(line).replace("'", '"')))
                else:
                    f_ann.write('{}\n'.format(str(line).replace("'", '"')))
    f_ann.close()
    f_ann_test.close()


if __name__ == '__main__':
    main()
