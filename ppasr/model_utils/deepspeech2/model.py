from paddle import nn

from ppasr.model_utils.deepspeech2.conv import ConvStack
from ppasr.model_utils.deepspeech2.rnn import RNNStack

__all__ = ['deepspeech2', 'deepspeech2_big']


class DeepSpeech2Model(nn.Layer):
    """DeepSpeech2模型结构

    :param feat_size: 输入的特征大小
    :type feat_size: int
    :param vocab_size: 字典的大小，用来分类输出
    :type vocab_size: int
    :param cnn_size: 卷积层的隐层大小
    :type cnn_size: int
    :param num_rnn_layers: 堆叠RNN层数
    :type num_rnn_layers: int
    :param rnn_size: RNN层大小
    :type rnn_size: int
    :param use_gru: 是否使用GRU，否则使用LSTM，大数据时LSTM效果会更好一些
    :type use_gru: bool

    :return: DeepSpeech2模型
    :rtype: nn.Layer
    """

    def __init__(self, feat_size, vocab_size, cnn_size=32, num_rnn_layers=5, rnn_size=1024, use_gru=True):
        super().__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_size = rnn_size
        # 卷积层堆
        self.conv = ConvStack(feat_size=feat_size, conv_out_channels=cnn_size)
        # RNN层堆
        self.rnn = RNNStack(i_size=self.conv.output_dim, h_size=rnn_size, num_rnn_layers=num_rnn_layers, use_gru=use_gru)
        # 分类输入层
        self.output = nn.Linear(self.rnn.output_dim, vocab_size)

    def forward(self, audio, audio_len, init_state_h_box=None, init_state_c_box=None):
        """
        Args:
            audio (Tensor): [B, D, Tmax]
            audio_len (Tensor): [B, Umax]
            init_state_h_box (Tensor): [num_rnn_layers, B, rnn_size]
        Returns:
            logits (Tensor): [B, T, D]
            x_lens (Tensor): [B]
        """
        # [B, T, D]
        x, x_lens = self.conv(audio, audio_len)
        # [B, T, D] [num_rnn_layers, B, rnn_size] [num_rnn_layers, B, rnn_size]
        x, final_chunk_state_h_box, final_chunk_state_c_box = self.rnn(x, x_lens, init_state_h_box, init_state_c_box)
        logits = self.output(x)
        if init_state_h_box is None:
            return logits, x_lens
        else:
            return logits, x_lens, final_chunk_state_h_box, final_chunk_state_c_box


# 获取普通的DeepSpeech模型
def deepspeech2(feat_size, vocab_size):
    model = DeepSpeech2Model(feat_size=feat_size,
                             vocab_size=vocab_size,
                             cnn_size=32,
                             num_rnn_layers=5,
                             rnn_size=1024,
                             use_gru=True)
    return model


# 获取大的DeepSpeech模型，适合训练Wenetspeech等大数据集
def deepspeech2_big(feat_size, vocab_size):
    model = DeepSpeech2Model(feat_size=feat_size,
                             vocab_size=vocab_size,
                             cnn_size=32,
                             num_rnn_layers=5,
                             rnn_size=2048,
                             use_gru=False)
    return model
