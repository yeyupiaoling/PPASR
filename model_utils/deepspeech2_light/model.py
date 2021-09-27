import math

import paddle
from paddle import ParamAttr, nn

from model_utils.deepspeech2_light.mobilenet import MobileNetV1
from model_utils.deepspeech2_light.rnn import BidirectionalGRU


def get_para_bias_attr(l2_decay, k):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)
    return [weight_attr, bias_attr]


class DeepSpeech2LightModel(nn.Layer):
    def __init__(self, vocab_size, rnn_size=128, scale=1.0):
        super().__init__()
        self.conv = MobileNetV1(scale)
        self.rnn = BidirectionalGRU(in_channels=self.conv.out_channels, hidden_size=128)
        weight_attr1, bias_attr1 = get_para_bias_attr(l2_decay=0.00002, k=self.rnn.out_channels)
        self.fc1 = nn.Linear(self.rnn.out_channels, rnn_size * 2, weight_attr=weight_attr1, bias_attr=bias_attr1)
        weight_attr2, bias_attr2 = get_para_bias_attr(l2_decay=0.00002, k=rnn_size * 2)
        self.fc2 = nn.Linear(rnn_size * 2, vocab_size, weight_attr=weight_attr2, bias_attr=bias_attr2)

    def forward(self, audio, audio_len):
        """
        Args:
            audio (Tensor): [B, D, Tmax]
            audio_len (Tensor): [B, Umax]
        Returns:
            logits (Tensor): [B, T, D]
            x_lens (Tensor): [B]
        """
        # [B, D, T] -> [B, C=1, D, T]
        x = audio.unsqueeze(1)
        x, x_len = self.conv(x, audio_len)  # [B, C, D, T]

        # 将数据从卷积特征映射转换为向量序列
        x = x.transpose([0, 3, 1, 2])  # [B, T, C, D]
        x = x.reshape([0, 0, -1])  # [B, T, C*D]

        x = self.rnn(x, x_len)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, x_len
