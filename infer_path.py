import argparse
import functools
import time

from data_utils.audio_process import AudioProcess
from utils.predict import Predictor
from utils.audio_vad import crop_audio_vad
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('wav_path',         str,    './dataset/test.wav', "预测音频的路径")
add_arg('is_long_audio',    bool,   False,  "是否为长语音")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('to_an',            bool,   True,   "是否转为阿拉伯数字")
add_arg('beam_size',        int,    10,     "集束搜索解码相关参数，搜索的大小，范围:[5, 500]")
add_arg('alpha',            float,  1.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  0.35,   "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  1.0,    "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('mean_std_path',    str,    'dataset/mean_std.npz',      "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',       str,    'dataset/vocabulary.txt',    "数据集的词汇表文件路径")
add_arg('model_dir',        str,    'models/infer/',             "导出的预测模型文件夹路径")
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',   "集束搜索解码相关参数，语言模型文件路径")
add_arg('decoder',          str,    'ctc_greedy',    "结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()
print_arguments(args)


# 提取音频特征器和归一化器
audio_process = AudioProcess(mean_std_filepath=args.mean_std_path, vocab_filepath=args.vocab_path)

predictor = Predictor(model_dir=args.model_dir, audio_process=audio_process, decoder=args.decoder,
                      alpha=args.alpha, beta=args.beta, lang_model_path=args.lang_model_path, beam_size=args.beam_size,
                      cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n, use_gpu=args.use_gpu)


def predict_long_audio():
    start = time.time()
    # 分割长音频
    audios_path = crop_audio_vad(args.wav_path)
    texts = ''
    scores = []
    # 执行识别
    for i, audio_path in enumerate(audios_path):
        score, text = predictor.predict(audio_path=audio_path, to_an=args.to_an)
        texts = texts + '，' + text
        scores.append(score)
        print("第%d个分割音频, 得分: %d, 识别结果: %s" % (i, score, text))
    print("最终结果，消耗时间：%d, 得分: %d, 识别结果: %s" % (round((time.time() - start) * 1000), sum(scores) / len(scores), texts))


def predict_audio():
    start = time.time()
    score, text = predictor.predict(audio_path=args.wav_path, to_an=args.to_an)
    print("消耗时间：%dms, 识别结果: %s, 得分: %d" % (round((time.time() - start) * 1000), text, score))


if __name__ == "__main__":
    if args.is_long_audio:
        predict_long_audio()
    else:
        predict_audio()
