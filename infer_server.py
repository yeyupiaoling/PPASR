import _thread
import argparse
import asyncio
import functools
import os
import time
import wave

import websockets
import yaml
from flask import request, Flask, render_template
from flask_cors import CORS

from ppasr.predict import Predictor
from ppasr.utils.audio_vad import crop_audio_vad
from ppasr.utils.utils import add_arguments, print_arguments
from ppasr.utils.logger import setup_logger

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/config_zh.yml', "配置文件")
add_arg("host",             str,    '0.0.0.0',            "监听主机的IP地址")
add_arg("port_server",      int,    5000,                 "普通识别服务所使用的端口号")
add_arg("port_stream",      int,    5001,                 "流式识别服务所使用的端口号")
add_arg("save_path",        str,    'dataset/upload/',    "上传音频文件的保存目录")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,  "是否对文本进行反标准化")
add_arg('num_predictor',    int,    1,      "多少个预测器，也是就可以同时有多少个用户同时识别")
add_arg('model_dir',        str,    'models/{}_{}/infer/',       "导出的预测模型文件夹路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',        "加标点符号的模型文件夹路径")
args = parser.parse_args()

# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)


app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# 允许跨越访问
CORS(app)

# 创建多个预测器，PaddlePaddle非线程安全，所以要这样处理
predictors = []
for _ in range(args.num_predictor):
    predictor1 = Predictor(configs=configs,
                           model_dir=args.model_dir.format(configs['use_model'],
                                                           configs['preprocess']['feature_method']),
                           use_gpu=args.use_gpu,
                           use_pun=args.use_pun,
                           pun_model_dir=args.pun_model_dir)
    predictors.append(predictor1)


# 语音识别接口
@app.route("/recognition", methods=['POST'])
def recognition():
    f = request.files['audio']
    # 可以让客户端自定义
    use_pun = args.use_pun
    # use_pun = request.form.get('use_pun')
    is_itn = args.is_itn
    # is_itn = request.form.get('is_itn')
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
                    score, text = predictor.predict(audio_path=file_path, use_pun=use_pun, is_itn=is_itn)
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
    is_itn = args.is_itn
    # is_itn = request.form.get('is_itn')
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
                        score, text = predictor.predict(audio_bytes=audio_bytes, use_pun=use_pun, is_itn=is_itn)
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


# 流式识别WebSocket服务
async def stream_server_run(websocket, path):
    logger.info(f'有WebSocket连接建立：{websocket.remote_address}')
    use_predictor = None
    for predictor in predictors:
        if predictor.running: continue
        use_predictor = predictor
        use_predictor.running = True
        break
    if use_predictor is not None:
        frames = []
        while not websocket.closed:
            try:
                data = await websocket.recv()
                frames.append(data)
                if len(data) == 0: continue
                is_end = False
                # 判断是不是结束预测
                if b'end' == data[-3:]:
                    is_end = True
                    data = data[:-3]
                # 开始预测
                score, text = use_predictor.predict_stream(audio_bytes=data, use_pun=args.use_pun, is_itn=args.is_itn,
                                                           is_end=is_end)
                send_data = str({"code": 0, "result": text}).replace("'", '"')
                logger.info(f'向客户端发生消息：{send_data}')
                await websocket.send(send_data)
                # 结束了要关闭当前的连接
                if is_end: await websocket.close()
            except Exception as e:
                logger.error(f'识别发生错误：错误信息：{e}')
                try:
                    await websocket.send(str({"code": 2, "msg": "recognition fail!"}))
                except:pass
        # 重置流式识别
        use_predictor.reset_stream()
        use_predictor.running = False
        # 保存录音
        save_path = os.path.join(args.save_path, f"{int(time.time() * 1000)}.wav")
        audio_bytes = b''.join(frames)
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
        wf.close()
    else:
        logger.error(f'语音识别失败，预测器不足')
        await websocket.send(str({"code": 1, "msg": "recognition fail, no resource!"}))
        websocket.close()


# 因为有多个服务需要使用线程启动
def start_server_thread():
    app.run(host=args.host, port=args.port_server)


if __name__ == '__main__':
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    _thread.start_new_thread(start_server_thread, ())
    # 启动Flask服务
    server = websockets.serve(stream_server_run, args.host, args.port_stream)
    # 启动WebSocket服务
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()

