import paddle
from paddle import nn

__all__ = ['Mask']


class Mask(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        """
        :param x: 卷积输入，shape[B, D, T]
        :param lengths: 卷积处理过的长度，shape[B]
        :return: 经过填充0的结果
        """
        batch_size = int(lengths.shape[0])
        max_len = int(lengths.max())
        seq_range = paddle.arange(0, max_len, dtype=paddle.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len])
        seq_length_expand = lengths.unsqueeze(-1).astype(paddle.int64)
        masks = paddle.less_than(seq_range_expand, seq_length_expand)
        masks = masks.astype(x.dtype)
        masks = masks.unsqueeze(1)  # [B, 1, T]
        x = x.multiply(masks)
        return x


class Normalizer(nn.Layer):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = paddle.to_tensor(mean, dtype=paddle.float32)
        self.std = paddle.to_tensor(std, dtype=paddle.float32)
        self.eps = 1e-20

    def forward(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x
