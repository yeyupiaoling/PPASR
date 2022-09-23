from paddle import nn

from ppasr.model_utils.deepspeech2_no_stream.conv import ConvStack
from ppasr.model_utils.deepspeech2_no_stream.rnn import RNNStack

__all__ = ['deepspeech2_no_stream', 'deepspeech2_big_no_stream']


class DeepSpeech2NoStreamModel(nn.Layer):
    """DeepSpeech2非流式模型结构
    :param feat_size: 输入的特征大小
    :type feat_size: int
    :param vocab_size: 字典的大小，用来分类输出
    :type vocab_size: int
    :param num_conv_layers: 堆叠卷积层数
    :type num_conv_layers: int
    :param num_rnn_layers: 堆叠RNN层数
    :type num_rnn_layers: int
    :param rnn_size: RNN层大小
    :type rnn_size: int
    :return: DeepSpeech2模型
    :rtype: nn.Layer
    """

    def __init__(self, feat_size, vocab_size, num_conv_layers=2, num_rnn_layers=3, rnn_size=1024, use_gru=True):
        super().__init__()
        # 卷积层堆
        self.conv = ConvStack(feat_size, num_conv_layers)
        # RNN层堆
        i_size = self.conv.output_height
        self.rnn = RNNStack(i_size=i_size, h_size=rnn_size, num_stacks=num_rnn_layers, use_gru=use_gru)
        # 分类输入层
        self.bn = nn.BatchNorm1D(rnn_size * 2, data_format='NLC')
        self.fc = nn.Linear(rnn_size * 2, vocab_size)

    def forward(self, audio, audio_len):
        """
        Args:
            audio (Tensor): [B, D, Tmax]
            audio_len (Tensor): [B, Umax]
        Returns:
            logits (Tensor): [B, T, D]
            x_lens (Tensor): [B]
        """
        # [B, T, D] -> [B, D, T]
        x = audio.transpose([0, 2, 1])
        # [B, D, T] -> [B, C=1, D, T]
        x = x.unsqueeze(1)

        x, x_lens = self.conv(x, audio_len)

        # 将数据从卷积特征映射转换为向量序列
        x = x.transpose([0, 3, 1, 2])  # [B, T, C, D]
        x = x.reshape([0, 0, -1])  # [B, T, C*D]
        x = self.rnn(x, x_lens)  # [B, T, D]

        x = self.bn(x)
        logits = self.fc(x)
        return logits, x_lens


def deepspeech2_no_stream(feat_size, vocab_size):
    model = DeepSpeech2NoStreamModel(feat_size=feat_size,
                                     vocab_size=vocab_size,
                                     num_conv_layers=2,
                                     num_rnn_layers=3,
                                     rnn_size=1024,
                                     use_gru=True)
    return model


def deepspeech2_big_no_stream(feat_size, vocab_size):
    model = DeepSpeech2NoStreamModel(feat_size=feat_size,
                                     vocab_size=vocab_size,
                                     num_conv_layers=2,
                                     num_rnn_layers=3,
                                     rnn_size=2048,
                                     use_gru=False)
    return model
