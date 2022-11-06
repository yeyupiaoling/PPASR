import _thread
import argparse
import asyncio
import functools
import json
import os
import time
import tkinter.messagebox
import wave
from tkinter import *
from tkinter.filedialog import askopenfilename

import pyaudio
import requests
import websockets
import yaml

from ppasr.predict import PPASRPredictor
from ppasr.utils.logger import setup_logger
from ppasr.utils.utils import add_arguments, print_arguments

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/conformer_online_zh.yml',       "配置文件")
add_arg('use_server',       bool,   False,         "是否使用服务器服务进行识别，否则使用本地识别")
add_arg("host",             str,    "127.0.0.1",   "服务器IP地址")
add_arg("port_server",      int,    5000,          "普通识别服务端口号")
add_arg("port_stream",      int,    5001,          "流式识别服务端口号")
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('model_path',       str,    'models/{}_{}/infer/',   "导出的预测模型文件路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',    "加标点符号的模型文件夹路径")
args = parser.parse_args()


# 读取配置文件
with open(args.configs, 'r', encoding='utf-8') as f:
    configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print_arguments(args, configs)


class SpeechRecognitionApp:
    def __init__(self, window: Tk, args):
        self.window = window
        self.wav_path = None
        self.predicting = False
        self.playing = False
        self.recording = False
        self.stream = None
        self.is_itn = False
        self.use_server = args.use_server
        # 录音参数
        self.frames = []
        interval_time = 0.5
        self.CHUNK = int(16000 * interval_time)
        # 最大录音时长
        self.max_record = 600
        # 录音保存的路径
        self.output_path = 'dataset/record'
        # 创建一个播放器
        self.p = pyaudio.PyAudio()
        # 指定窗口标题
        self.window.title("夜雨飘零语音识别")
        # 固定窗口大小
        self.window.geometry('870x500')
        self.window.resizable(False, False)
        # 识别短语音按钮
        self.short_button = Button(self.window, text="选择短语音识别", width=20, command=self.predict_audio_thread)
        self.short_button.place(x=10, y=10)
        # 识别长语音按钮
        self.long_button = Button(self.window, text="选择长语音识别", width=20, command=self.predict_long_audio_thread)
        self.long_button.place(x=170, y=10)
        # 录音按钮
        self.record_button = Button(self.window, text="录音识别", width=20, command=self.record_audio_thread)
        self.record_button.place(x=330, y=10)
        # 播放音频按钮
        self.play_button = Button(self.window, text="播放音频", width=20, command=self.play_audio_thread)
        self.play_button.place(x=490, y=10)
        # 输出结果文本框
        self.result_label = Label(self.window, text="输出日志：")
        self.result_label.place(x=10, y=70)
        self.result_text = Text(self.window, width=120, height=30)
        self.result_text.place(x=10, y=100)
        # 对文本进行反标准化
        self.an_frame = Frame(self.window)
        self.check_var = BooleanVar(value=False)
        self.is_itn_check = Checkbutton(self.an_frame, text='是否对文本进行反标准化', variable=self.check_var, command=self.is_itn_state)
        self.is_itn_check.grid(row=0)
        self.an_frame.grid(row=1)
        self.an_frame.place(x=700, y=10)

        if not self.use_server:
            # 获取识别器
            self.predictor = PPASRPredictor(configs=configs,
                                            model_path=args.model_path.format(configs['use_model'],
                                                                              configs['preprocess_conf']['feature_method']),
                                            use_gpu=args.use_gpu,
                                            use_pun=args.use_pun,
                                            pun_model_dir=args.pun_model_dir)

    # 是否对文本进行反标准化
    def is_itn_state(self):
        self.is_itn = self.check_var.get()

    # 预测短语音线程
    def predict_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir='./dataset')
            if self.wav_path == '': return
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, "已选择音频文件：%s\n" % self.wav_path)
            self.result_text.insert(END, "正在识别中...\n")
            _thread.start_new_thread(self.predict_audio, (self.wav_path, ))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测短语音
    def predict_audio(self, wav_file):
        self.predicting = True
        try:
            start = time.time()
            # 判断使用本地识别还是调用服务接口
            if not self.use_server:
                result = self.predictor.predict(audio_data=wav_file, use_pun=args.use_pun, is_itn=self.is_itn)
                score, text = result['score'], result['text']
            else:
                # 调用用服务接口识别
                url = f"http://{args.host}:{args.port_server}/recognition"
                files = [('audio', ('test.wav', open(wav_file, 'rb'), 'audio/wav'))]
                headers = {'accept': 'application/json'}
                response = requests.post(url, headers=headers, files=files)
                data = json.loads(response.text)
                if data['code'] != 0:
                    raise Exception(f'服务请求失败，错误信息：{data["msg"]}')
                text, score = data['result'], data['score']
            self.result_text.insert(END, f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {text}, 得分: {score}\n")
        except Exception as e:
            self.result_text.insert(END, str(e))
            logger.error(e)
        self.predicting = False

    # 预测长语音线程
    def predict_long_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir='./dataset')
            if self.wav_path == '': return
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, "已选择音频文件：%s\n" % self.wav_path)
            self.result_text.insert(END, "正在识别中...\n")
            _thread.start_new_thread(self.predict_long_audio, (self.wav_path, ))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测长语音
    def predict_long_audio(self, wav_path):
        self.predicting = True
        try:
            start = time.time()
            # 判断使用本地识别还是调用服务接口
            if not self.use_server:
                result = self.predictor.predict_long(audio_data=wav_path, use_pun=args.use_pun, is_itn=self.is_itn)
                score, text = result['score'], result['text']
            else:
                # 调用用服务接口识别
                url = f"http://{args.host}:{args.port_server}/recognition_long_audio"
                files = [('audio', ('test.wav', open(wav_path, 'rb'), 'audio/wav'))]
                headers = {'accept': 'application/json'}
                response = requests.post(url, headers=headers, files=files)
                data = json.loads(response.text)
                if data['code'] != 0:
                    raise Exception(f'服务请求失败，错误信息：{data["msg"]}')
                text, score = data['result'], data['score']
            self.result_text.insert(END, "=====================================================\n")
            self.result_text.insert(END, f"最终结果，消耗时间：{int(round((time.time() - start) * 1000))}, 得分: {score}, 识别结果: {text}\n")
        except Exception as e:
            self.result_text.insert(END, str(e))
            logger.error(e)
        self.predicting = False

    # 录音识别线程
    def record_audio_thread(self):
        if not self.playing and not self.recording:
            self.result_text.delete('1.0', 'end')
            _thread.start_new_thread(self.record_audio, ())
        else:
            if self.playing:
                tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
            else:
                # 停止播放
                self.recording = False

    # 使用WebSocket调用实时语音识别服务
    async def run_websocket(self):
        async with websockets.connect(f"ws://{args.host}:{args.port_stream}") as websocket:
            while not websocket.closed:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
                send_data = data
                # 用户点击停止录音按钮
                if not self.recording:
                    send_data += b'end'
                await websocket.send(send_data)
                result = await websocket.recv()
                self.result_text.delete('1.0', 'end')
                self.result_text.insert(END, f"{json.loads(result)['result']}\n")
                # 停止录音后，需要把end发给服务器才能最终停止
                if not self.recording and b'end' == send_data[-3:]:break
            # await websocket.close()
        logger.info('close websocket')

    def record_audio(self):
        self.record_button.configure(text='停止录音')
        self.recording = True
        self.frames = []
        FORMAT = pyaudio.paInt16
        channels = 1
        rate = 16000

        # 打开录音
        self.stream = self.p.open(format=FORMAT,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        self.result_text.insert(END, "正在录音...\n")
        if not self.use_server:
            # 本地识别
            while True:
                data = self.stream.read(self.CHUNK)
                self.frames.append(data)
                result = self.predictor.predict_stream(audio_data=data, use_pun=args.use_pun, is_itn=self.is_itn, is_end=not self.recording)
                if result is None:continue
                score, text = result['score'], result['text']
                self.result_text.delete('1.0', 'end')
                self.result_text.insert(END, f"{text}\n")
                if not self.recording:break
            self.predictor.reset_stream()
        else:
            # 调用服务接口
            new_loop = asyncio.new_event_loop()
            new_loop.run_until_complete(self.run_websocket())

        # 录音的字节数据，用于后面的预测和保存
        audio_bytes = b''.join(self.frames)
        # 保存音频数据
        os.makedirs(self.output_path, exist_ok=True)
        self.wav_path = os.path.join(self.output_path, '%s.wav' % str(int(time.time())))
        wf = wave.open(self.wav_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(audio_bytes)
        wf.close()
        self.recording = False
        self.result_text.insert(END, "录音已结束，录音文件保存在：%s\n" % self.wav_path)
        self.record_button.configure(text='录音识别')

    # 播放音频线程
    def play_audio_thread(self):
        if self.wav_path is None or self.wav_path == '':
            tkinter.messagebox.showwarning('警告', '音频路径为空！')
        else:
            if not self.playing and not self.recording:
                _thread.start_new_thread(self.play_audio, ())
            else:
                if self.recording:
                    tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
                else:
                    # 停止播放
                    self.playing = False

    # 播放音频
    def play_audio(self):
        self.play_button.configure(text='停止播放')
        self.playing = True
        CHUNK = 1024
        wf = wave.open(self.wav_path, 'rb')
        # 打开数据流
        self.stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                  channels=wf.getnchannels(),
                                  rate=wf.getframerate(),
                                  output=True)
        # 读取数据
        data = wf.readframes(CHUNK)
        # 播放
        while data != b'':
            if not self.playing:break
            self.stream.write(data)
            data = wf.readframes(CHUNK)
        # 停止数据流
        self.stream.stop_stream()
        self.stream.close()
        self.playing = False
        self.play_button.configure(text='播放音频')


tk = Tk()
myapp = SpeechRecognitionApp(tk, args)

if __name__ == '__main__':
    tk.mainloop()
