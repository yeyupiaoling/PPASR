import paddle
import os
from utils.decoder import GreedyDecoder
from utils.model import PPASR
from utils.data import load_audio, audio_to_stft

dict_path = "dataset/zh_vocab.json"
model_path = 'models/epoch_3/model.pdparams'
audio_path = 'dataset/test.wav'

# 加载数据字典
with open(dict_path) as f:
    labels = eval(f.read())
vocabulary = dict([(labels[i], i) for i in range(len(labels))])
greedy_decoder = GreedyDecoder(vocabulary)

model = PPASR(vocabulary)
model.set_state_dict(paddle.load(model_path))
model.eval()

wav = load_audio(audio_path)
stft = audio_to_stft(wav)

stft = paddle.to_tensor(stft, dtype='float32')
print(stft)
stft = paddle.unsqueeze(stft, axis=0)
out = model(stft)
out = paddle.nn.functional.softmax(out, 1)
print(out)
print(out.shape)
out = paddle.transpose(out, perm=[0, 2, 1])
print(out)
out_strings, out_offsets = greedy_decoder.decode(out)
print(out_strings)