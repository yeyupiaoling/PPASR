import _thread
import argparse
import functools
import os
import time
import tkinter.messagebox
import wave
from tkinter import *
from tkinter.filedialog import askopenfilename

import pyaudio

from ppasr import SUPPORT_MODEL
from ppasr.predict import Predictor
from ppasr.utils.audio_vad import crop_audio_vad
from ppasr.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'deepspeech2', '所使用的模型', choices=SUPPORT_MODEL)
add_arg('use_gpu',          bool,   True,   "是否使用GPU预测")
add_arg('use_server',       bool,   True,   "是否使用GPU预测")
add_arg('use_pun',          bool,   False,  "是否给识别结果加标点符号")
add_arg('beam_size',        int,    300,    "集束搜索解码相关参数，搜索的大小，范围:[5, 500]")
add_arg('alpha',            float,  2.2,    "集束搜索解码相关参数，LM系数")
add_arg('beta',             float,  4.3,    "集束搜索解码相关参数，WC系数")
add_arg('cutoff_prob',      float,  0.99,   "集束搜索解码相关参数，剪枝的概率")
add_arg('cutoff_top_n',     int,    40,     "集束搜索解码相关参数，剪枝的最大值")
add_arg('vocab_path',       str,    'dataset/vocabulary.txt',    "数据集的词汇表文件路径")
add_arg('model_dir',        str,    'models/{}_{}/infer/',       "导出的预测模型文件夹路径")
add_arg('pun_model_dir',    str,    'models/pun_models/',        "加标点符号的模型文件夹路径")
add_arg('lang_model_path',  str,    'lm/zh_giga.no_cna_cmn.prune01244.klm',   "集束搜索解码相关参数，语言模型文件路径")
add_arg('feature_method',   str,    'linear',             "音频预处理方法", choices=['linear', 'mfcc', 'fbank'])
add_arg('decoder',          str,    'ctc_beam_search',    "结果解码方法",   choices=['ctc_beam_search', 'ctc_greedy'])
args = parser.parse_args()
print_arguments(args)


class SpeechRecognitionApp:
    def __init__(self, window: Tk, args):
        self.window = window
        self.wav_path = None
        self.predicting = False
        self.playing = False
        self.recording = False
        self.stream = None
        self.to_an = False
        self.use_server = args.use_server
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
        # 转阿拉伯数字控件
        self.an_frame = Frame(self.window)
        self.check_var = BooleanVar()
        self.to_an_check = Checkbutton(self.an_frame, text='中文数字转阿拉伯数字', variable=self.check_var, command=self.to_an_state)
        self.to_an_check.grid(row=0)
        self.an_frame.grid(row=1)
        self.an_frame.place(x=700, y=10)

        if not self.use_server:
            # 获取识别器
            self.predictor = Predictor(model_dir=args.model_dir.format(args.use_model, args.feature_method), vocab_path=args.vocab_path,
                                       use_model=args.use_model, decoder=args.decoder, alpha=args.alpha, beta=args.beta,
                                       lang_model_path=args.lang_model_path,
                                       beam_size=args.beam_size, cutoff_prob=args.cutoff_prob,
                                       cutoff_top_n=args.cutoff_top_n, feature_method=args.feature_method,
                                       use_gpu=args.use_gpu, use_pun=args.use_pun, pun_model_dir=args.pun_model_dir)

    # 是否中文数字转阿拉伯数字
    def to_an_state(self):
        self.to_an = self.check_var.get()

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
            if isinstance(wav_file, str):
                score, text = self.predictor.predict(audio_path=wav_file, use_pun=args.use_pun, to_an=self.to_an)
            else:
                score, text = self.predictor.predict(audio_bytes=wav_file, use_pun=args.use_pun, to_an=self.to_an)
            self.result_text.insert(END, "消耗时间：%dms, 识别结果: %s, 得分: %d\n" % (
            round((time.time() - start) * 1000), text, score))
        except Exception as e:
            print(e)
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
            # 分割长音频
            audios_bytes = crop_audio_vad(wav_path)
            texts = ''
            scores = []
            # 执行识别
            for i, audio_bytes in enumerate(audios_bytes):
                score, text = self.predictor.predict(audio_bytes=audio_bytes, use_pun=args.use_pun, to_an=self.to_an)
                texts = texts + text if args.use_pun else texts + '，' + text
                scores.append(score)
                self.result_text.insert(END, "第%d个分割音频, 得分: %d, 识别结果: %s\n" % (i, score, text))
            self.result_text.insert(END, "=====================================================\n")
            self.result_text.insert(END, "最终结果，消耗时间：%d, 得分: %d, 识别结果: %s\n" %
                                    (round((time.time() - start) * 1000), sum(scores) / len(scores), texts))
        except Exception as e:
            print(e)
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

    def record_audio(self):
        self.record_button.configure(text='停止录音')
        self.recording = True
        # 录音参数
        interval_time = 0.5
        CHUNK = int(16000 * interval_time)
        FORMAT = pyaudio.paInt16
        channels = 1
        rate = 16000

        # 打开录音
        self.stream = self.p.open(format=FORMAT,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=CHUNK)
        self.result_text.insert(END, "正在录音...\n")
        frames, result = [], []
        while True:
            data = self.stream.read(CHUNK)
            frames.append(data)
            score, text = self.predictor.predict_stream(audio_bytes=data, use_pun=args.use_pun, to_an=self.to_an, is_end=not self.recording)
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, f"{text}\n")
            if not self.recording:break

        self.predictor.reset_stream()
        # 录音的字节数据，用于后面的预测和保存
        audio_bytes = b''.join(frames)
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
