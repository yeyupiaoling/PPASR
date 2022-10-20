import paddle
from paddle import nn


class Conv2dSubsampling4Pure(nn.Layer):
    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.subsampling_rate = 4
        self.output_dim = ((idim - 1) // 2 - 1) // 2 * odim

    def forward(self, x: paddle.Tensor, x_len: paddle.Tensor):
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        x = x.transpose([0, 2, 1, 3]).reshape([0, 0, -1])
        x_len = ((x_len - 1) // 2 - 1) // 2
        return x, x_len
