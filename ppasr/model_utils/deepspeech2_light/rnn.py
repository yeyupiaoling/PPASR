import paddle
from paddle import nn


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


class BidirectionalGRU(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        super(BidirectionalGRU, self).__init__()
        self.out_channels = hidden_size * 2
        self.lstm = nn.GRU(in_channels, hidden_size, direction='bidirectional', num_layers=2)
        self.mask = MaskRNN()

    def forward(self, x, x_len):
        x, _ = self.lstm(x)
        x = self.mask(x, x_len)
        return x
