import argparse
import functools
import os
import time

from flask import request, Flask, render_template
from flask_cors import CORS

from ppasr import SUPPORT_MODEL
from ppasr.predict import Predictor
from ppasr.utils.audio_vad import crop_audio_vad
from ppasr.utils.utils import add_arguments, print_arguments
from ppasr.utils.logger import setup_logger

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'deepspeech2',        "所使用的模型", choices=SUPPORT_MODEL)
add_arg("host",             str,    "0.0.0.0",            "监听主机的IP地址")
add_arg("port",             int,    5000,                 "服务所使用的端口号")
add_arg("save_path",        str,    'dataset/upload/',    "上传音频文件的保存目录")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('to_an',            bool,   False,  "是否转为阿拉伯数字")
add_arg('num_predictor',    int,    1,      "多少个预测器，也是就可以同时有多少个用户同时识别")
add_arg('beam_size',        int,    300,    "集束搜索解码相关参数，搜索大小，范围:[5, 500]")
add_arg('alpha',            float,  2.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  4.3,    "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  0.99,   "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('vocab_path',       str,    'dataset/vocabulary.txt',    "数据集的词汇表文件路径")
add_arg('model_dir',        str,    'models/{}_{}/infer/',       "导出的预测模型文件夹路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',        "加标点符号的模型文件夹路径")
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',    "集束搜索解码相关参数，语言模型文件路径")
add_arg('feature_method',   str,    'linear',             "音频预处理方法", choices=['linear', 'mfcc', 'fbank'])
add_arg('decoder',          str,    'ctc_beam_search',    "结果解码方法", choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# 允许跨越访问
CORS(app)

# 创建多个预测器，PaddlePaddle非线程安全，所以要这样处理
predictors = []
for _ in range(args.num_predictor):
    predictor1 = Predictor(model_dir=args.model_dir.format(args.use_model, args.feature_method), vocab_path=args.vocab_path, use_model=args.use_model,
                           decoder=args.decoder, alpha=args.alpha, beta=args.beta, lang_model_path=args.lang_model_path,
                           beam_size=args.beam_size, cutoff_prob=args.cutoff_prob, cutoff_top_n=args.cutoff_top_n,
                           use_gpu=args.use_gpu, use_pun=args.use_pun, pun_model_dir=args.pun_model_dir,
                           feature_method=args.feature_method)
    predictors.append(predictor1)


# 语音识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    f = request.files['audio']
    # 可以让客户端自定义
    use_pun = args.use_pun
    # use_pun = request.form.get('use_pun')
    to_an = args.to_an
    # to_an = request.form.get('to_an')
    if f:
        # 临时保存路径
        file_path = os.path.join(args.save_path, f"{int(time.time() * 1000)}.{f.filename.split('.')[-1]}")
        f.save(file_path)
        try:
            start = time.time()
            score, text = None, None
            # PaddlePaddle非线程安全，所以要这样处理
            for predictor in predictors:
                if predictor.running:continue
                predictor.running = True
                try:
                    score, text = predictor.predict(audio_path=file_path, use_pun=use_pun, to_an=to_an)
                except Exception as e:
                    predictor.running = False
                    raise Exception(e)
                predictor.running = False
                break
            if score is None and text is None:
                logger.error(f'短语音识别失败，预测器不足')
                return str({"error": 4, "msg": "recognition fail, no resource!"})
            end = time.time()
            logger.info(f"识别时间：{round((end - start) * 1000)}ms，识别结果：{text}， 得分: {score}")
            result = str({"code": 0, "msg": "success", "result": text, "score": round(score, 3)}).replace("'", '"')
            return result
        except Exception as e:
            logger.error(f'短语音识别失败，错误信息：{e}')
            return str({"error": 1, "msg": "audio read fail!"})
    return str({"error": 3, "msg": "audio is None!"})


# 长语音识别接口
@app.route("/recognition_long_audio", methods=['POST'])
def recognition_long_audio():
    f = request.files['audio']
    # 可以让客户端自定义
    use_pun = args.use_pun
    # use_pun = request.form.get('use_pun')
    to_an = args.to_an
    # to_an = request.form.get('to_an')
    if f:
        # 临时保存路径
        file_path = os.path.join(args.save_path, f"{int(time.time() * 1000)}.{f.filename.split('.')[-1]}")
        f.save(file_path)
        try:
            start = time.time()
            # 分割长音频
            audios_bytes = crop_audio_vad(file_path)
            texts = ''
            scores = []
            # PaddlePaddle非线程安全，所以要这样处理
            for predictor in predictors:
                if predictor.running:continue
                predictor.running = True
                try:
                    # 执行识别
                    for i, audio_bytes in enumerate(audios_bytes):
                        score, text = predictor.predict(audio_bytes=audio_bytes, use_pun=use_pun, to_an=to_an)
                        texts = texts + text if args.use_pun else texts + '，' + text
                        scores.append(score)
                except Exception as e:
                    predictor.running = False
                    raise Exception(e)
                predictor.running = False
                break
            if len(scores) == 0:
                logger.error(f'短语音识别失败，预测器不足')
                return str({"error": 4, "msg": "recognition fail, no resource!"})
            end = time.time()
            logger.info(f"识别时间：{round((end - start) * 1000)}ms，识别结果：{texts}, 得分: {sum(scores) / len(scores)}")
            result = str({"code": 0, "msg": "success", "result": texts, "score": round(float(sum(scores) / len(scores)), 3)}).replace("'", '"')
            return result
        except Exception as e:
            logger.error(f'短语音识别失败，错误信息：{e}')
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
