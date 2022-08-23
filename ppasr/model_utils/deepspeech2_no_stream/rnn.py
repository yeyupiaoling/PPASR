import paddle
from paddle import nn

__all__ = ['RNNStack']


class MaskRNN(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        """
        :param x: RNN输入，shape[B, T, D]
        :param lengths: RNN处理过的长度，shape[B]
        :return: 经过填充0的结果
        """
        batch_size = int(lengths.shape[0])
        max_len = int(lengths.max())
        seq_range = paddle.arange(0, max_len, dtype=paddle.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len])
        seq_length_expand = lengths.unsqueeze(-1).astype(paddle.int64)
        masks = paddle.less_than(seq_range_expand, seq_length_expand)
        masks = masks.astype(x.dtype)
        masks = masks.unsqueeze(-1)  # [B, T, 1]
        x = x.multiply(masks)
        return x


class BiRNNWithBN(nn.Layer):
    """具有顺序批标准化的双向RNN层。批标准化只对输入状态权值执行。
    :param i_size: GRUCell的输入大小
    :type i_size: int
    :param h_size: GRUCell的隐藏大小
    :type h_size: string
    :return: 双向RNN层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, use_gru:bool):
        super().__init__()
        hidden_size = h_size * 3
        self.mask = MaskRNN()

        self.fc = nn.Linear(i_size, hidden_size)
        self.bn = nn.BatchNorm1D(hidden_size, data_format='NLC')
        if use_gru:
            self.gru = nn.GRU(input_size=hidden_size, hidden_size=h_size, direction='bidirectional')
        else:
            self.gru = nn.LSTM(input_size=hidden_size, hidden_size=h_size, direction='bidirectional')

    def forward(self, x, x_len):
        # x, shape [B, T, D]
        x = self.bn(self.fc(x))
        x, _ = self.gru(inputs=x, sequence_length=x_len)

        # 将填充部分重置为0
        x = self.mask(x, x_len)
        return x


class RNNStack(nn.Layer):
    """RNN组与堆叠双向简单RNN或GRU层
    :param i_size: GRU层的输入大小
    :type i_size: int
    :param h_size: GRU层的隐层大小
    :type h_size: int
    :param num_stacks: 堆叠的rnn层数
    :type num_stacks: int
    :return: RNN组的输出层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, num_stacks: int, use_gru:bool):
        super().__init__()
        rnn_stacks = []
        for i in range(num_stacks):
            rnn_stacks.append(BiRNNWithBN(i_size=i_size, h_size=h_size, use_gru=use_gru))
            i_size = h_size * 2

        self.rnn_stacks = nn.LayerList(rnn_stacks)

    def forward(self, x: paddle.Tensor, x_len: paddle.Tensor):
        """
        x: shape [B, T, D]
        x_len: shpae [B]
        """
        for i, rnn in enumerate(self.rnn_stacks):
            x = rnn(x, x_len)
        return x