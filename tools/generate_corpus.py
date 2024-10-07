import os
import re

import cn2an


# 判断是否为中文字符
def is_uchar(in_str):
    for i in range(len(in_str)):
        uchar = in_str[i]
        if u'\u4e00' <= uchar <= u'\u9fa5':
            pass
        else:
            return False
    return True


# 制作中文语料
utt_id = 0
corpus_dir = 'dgk_lost_conv/results/'
with open('corpus.txt', 'w', encoding='utf-8') as f_write:
    for corpus_path in os.listdir(corpus_dir):
        if corpus_path[-5:] != '.conv': continue
        corpus_path = os.path.join(corpus_dir, corpus_path)
        print(corpus_path)
        if 'dgk_shooter_z.conv' in corpus_path:
            lines = []
            with open(corpus_path, 'r', encoding='utf-8') as f:
                while True:
                    try:
                        line = f.readline().replace('\n', '')
                        lines.append(line)
                        if len(line) == 0: break
                    except:
                        continue
        else:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        for line in lines:
            line = line[2:].replace('/', '').replace('\n', '').replace('?', '？').replace(' ', '%').replace('.', '')
            line = line.replace('～', '！').replace(',', '，').replace('、', '，').replace('!', '！').replace('"', '')
            line = line.replace('，，', '，').replace('。。', '。').replace('！！', '！').replace('？？', '？')
            line = line.replace('，，', '，').replace('。。', '。').replace('！！', '！').replace('？？', '？')
            line = cn2an.transform(line, "an2cn")
            if len(line) < 2: continue
            if not is_uchar(line.replace('，', '').replace('。', '').replace('？', '').replace('！', '')): continue
            my_re = re.compile(r'[A-Za-z0-9]', re.S)
            res = re.findall(my_re, line)
            if len(res) > 0: continue
            f_write.write('%d %s\n' % (utt_id, line))
            utt_id += 1