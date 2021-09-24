import argparse
import functools
import os
import sys
import time

from flask import request, Flask, render_template
from flask_cors import CORS

from data_utils.audio_process import AudioProcess
from utils.predict import Predictor
from utils.audio_vad import crop_audio_vad
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("host",             str,    "0.0.0.0",            "监听主机的IP地址")
add_arg("port",             int,    5000,                 "服务所使用的端口号")
add_arg("save_path",        str,    'dataset/upload/',    "上传音频文件的保存目录")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('to_an',            bool,   True,   "是否转为阿拉伯数字")
add_arg('beam_size',        int,    10,     "集束搜索解码相关参数，搜索大小，范围:[5, 500]")
add_arg('alpha',            float,  1.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  0.35,   "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  1.0,    "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('mean_std_path',    str,    'dataset/mean_std.npz',      "数据集的均值和标准值的npy文件路径")
add_arg('vocab_path',       str,    'dataset/vocabulary.txt',    "数据集的词汇表文件路径")
add_arg('model_dir',        str,    'models/deepspeech2/infer/', "导出的预测模型文件夹路径")
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',    "集束搜索解码相关参数，语言模型文件路径")
add_arg('decoder',          str,    'ctc_greedy',    "结果解码方法，有集束搜索(ctc_beam_search)、贪婪策略(ctc_greedy)", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# 允许跨越访问
CORS(app)

# 提取音频特征器和归一化器
audio_process = AudioProcess(mean_std_filepath=args.mean_std_path, vocab_filepath=args.vocab_path)

predictor = Predictor(model_dir=args.model_dir, audio_process=audio_process, decoder=args.decoder,
                      alpha=args.alpha, beta=args.beta, lang_model_path=args.lang_model_path, beam_size=args.beam_size,
                      cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n, use_gpu=args.use_gpu)


# 语音识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    f = request.files['audio']
    if f:
        # 临时保存路径
        file_path = os.path.join(args.save_path, f.filename)
        f.save(file_path)
        try:
            start = time.time()
            # 执行识别
            score, text = predictor.predict(audio_path=file_path, to_an=args.to_an)
            end = time.time()
            print("识别时间：%dms，识别结果：%s， 得分: %f" % (round((end - start) * 1000), text, score))
            result = str({"code": 0, "msg": "success", "result": text, "score": round(score, 3)}).replace("'", '"')
            return result
        except:
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


# 长语音识别接口
@app.route("/recognition_long_audio", methods=['POST'])
def recognition_long_audio():
    f = request.files['audio']
    if f:
        # 临时保存路径
        file_path = os.path.join(args.save_path, f.filename)
        f.save(file_path)
        try:
            start = time.time()
            # 分割长音频
            audios_path = crop_audio_vad(file_path)
            texts = ''
            scores = []
            # 执行识别
            for i, audio_path in enumerate(audios_path):
                score, text = predictor.predict(audio_path=audio_path, to_an=args.to_an)
                texts = texts + '，' + text
                scores.append(score)
            end = time.time()
            print("识别时间：%dms，识别结果：%s， 得分: %f" % (round((end - start) * 1000), texts, sum(scores) / len(scores)))
            result = str({"code": 0, "msg": "success", "result": texts, "score": round(float(sum(scores) / len(scores)), 3)}).replace("'", '"')
            return result
        except Exception as e:
            print(e, file=sys.stderr)
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


@app.route('/')
def home():
    return render_template("index.html")


if __name__ == '__main__':
    print_arguments(args)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    app.run(host=args.host, port=args.port)
