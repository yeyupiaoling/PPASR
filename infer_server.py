import argparse
import functools
import os
import random
import sys
import time
import wave
from datetime import datetime

import aiofiles
import uvicorn
from fastapi import FastAPI, WebSocket, UploadFile, File
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.websockets import WebSocketState

from ppasr.predict import PPASRPredictor
from ppasr.utils.logger import setup_logger
from ppasr.utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/conformer.yml', "配置文件")
add_arg("host",             str,    '0.0.0.0',            "监听主机的IP地址")
add_arg("port",             int,    5000,                 "服务所使用的端口号")
add_arg("save_path",        str,    'dataset/upload/',    "上传音频文件的保存目录")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('is_itn',           bool,   False,  "是否对文本进行反标准化")
add_arg('model_path',       str,    'models/conformer_streaming_fbank/infer',   "导出的预测模型文件路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',    "加标点符号的模型文件夹路径")
args = parser.parse_args()
print_arguments(args=args)

app = FastAPI(title="PPASR")
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory="templates")

# 创建预测器，是实时语音的第一个对象和创建多进程时使用
predictor = PPASRPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu,
                           use_pun=args.use_pun,
                           pun_model_dir=args.pun_model_dir)


# 语音识别接口
@app.post("/recognition")
async def recognition(audio: UploadFile = File(..., description="音频文件")):
    # 保存路径
    save_dir = os.path.join(args.save_path, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(save_dir, exist_ok=True)
    suffix = audio.filename.split('.')[-1]
    file_path = os.path.join(save_dir, f'{int(time.time() * 1000)}_{random.randint(100, 999)}.{suffix}')
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        start = time.time()
        # 执行识别
        result = predictor.predict(audio_data=file_path, use_pun=args.use_pun, is_itn=args.is_itn)
        score, text = result['score'], result['text']
        end = time.time()
        print("识别时间：%dms，识别结果：%s， 得分: %f" % (round((end - start) * 1000), text, score))
        result = str({"code": 0, "msg": "success", "result": text, "score": round(score, 3)}).replace("'", '"')
        return result
    except Exception as e:
        print(f'[{datetime.now()}] 短语音识别失败，错误信息：{e}', file=sys.stderr)
        return str({"error": 1, "msg": "audio read fail!"})


# 长语音识别接口
@app.post("/recognition_long_audio")
async def recognition_long_audio(audio: UploadFile = File(..., description="音频文件")):
    # 保存路径
    save_dir = os.path.join(args.save_path, datetime.now().strftime('%Y-%m-%d'))
    os.makedirs(save_dir, exist_ok=True)
    suffix = audio.filename.split('.')[-1]
    file_path = os.path.join(save_dir, f'{int(time.time() * 1000)}_{random.randint(100, 999)}.{suffix}')
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        start = time.time()
        result = predictor.predict_long(audio_data=file_path, use_pun=args.use_pun, is_itn=args.is_itn)
        score, text = result['score'], result['text']
        end = time.time()
        print("识别时间：%dms，识别结果：%s， 得分: %f" % (round((end - start) * 1000), text, score))
        result = str({"code": 0, "msg": "success", "result": text, "score": score}).replace("'", '"')
        return result
    except Exception as e:
        print(f'[{datetime.now()}] 长语音识别失败，错误信息：{e}', file=sys.stderr)
        return str({"error": 1, "msg": "audio read fail!"})


@app.get("/")
async def index():
    return templates.TemplateResponse("index.html")


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f'有WebSocket连接建立')
    if not predictor.running:
        frames = []
        score, text = 0, ""
        while True:
            try:
                data = await websocket.receive_bytes()
                frames.append(data)
                if len(data) == 0: continue
                is_end = False
                # 判断是不是结束预测
                if b'end' == data[-3:]:
                    is_end = True
                    data = data[:-3]
                # 开始预测
                result = predictor.predict_stream(audio_data=data, use_pun=args.use_pun, is_itn=args.is_itn,
                                                  is_end=is_end)
                if result is not None:
                    score, text = result['score'], result['text']
                send_data = {"code": 0, "result": text}
                logger.info(f'向客户端发生消息：{send_data}')
                await websocket.send_json(send_data)
                # 结束了要关闭当前的连接
                if is_end: await websocket.close()
            except Exception as e:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("用户已断开连接")
                    break
                logger.error(f'识别发生错误：错误信息：{e}')
                try:
                    await websocket.send_json({"code": 2, "msg": "recognition fail!"})
                except:
                    break
        # 重置流式识别
        predictor.reset_stream()
        predictor.running = False
        # 保存录音
        save_dir = os.path.join(args.save_path, datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{int(time.time() * 1000)}.wav')
        audio_bytes = b''.join(frames)
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
        wf.close()
    else:
        logger.error(f'语音识别失败，预测器不足')
        await websocket.send_json({"code": 1, "msg": "recognition fail, no resource!"})
        await websocket.close()


if __name__ == '__main__':
    # 创建保存路径
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    uvicorn.run(app, host=args.host, port=args.port)
