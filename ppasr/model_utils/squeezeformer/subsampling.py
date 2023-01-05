from typing import Tuple

import paddle
import paddle.nn as nn

from ppasr.model_utils.conformer.subsampling import BaseSubsampling
from ppasr.model_utils.utils.base import Conv2D, Linear

__all__ = ["DepthwiseConv2DSubsampling4"]


class DepthwiseConv2DSubsampling4(BaseSubsampling):
    """Depthwise Convolutional 2D subsampling (to 1/4 length).

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            pos_enc_class (nn.Layer): position encoding class.
            dw_stride (int): Whether do depthwise convolution.
            input_size (int): filter bank dimension.

        """

    def __init__(self,
                 idim: int,
                 odim: int,
                 pos_enc_class: nn.Layer,
                 dw_stride: bool = False,
                 input_size: int = 80,
                 input_dropout_rate: float = 0.1,
                 init_weights: bool = True):
        super(DepthwiseConv2DSubsampling4, self).__init__()
        self.idim = idim
        self.odim = odim
        self.pw_conv = Conv2D(in_channels=idim, out_channels=odim, kernel_size=3, stride=2)
        self.act1 = nn.ReLU()
        self.dw_conv = Conv2D(in_channels=odim, out_channels=odim, kernel_size=3, stride=2,
                              groups=odim if dw_stride else 1)
        self.act2 = nn.ReLU()
        self.pos_enc = pos_enc_class
        self.input_proj = nn.Sequential(
            Linear(odim * (((input_size - 1) // 2 - 1) // 2), odim),
            nn.Dropout(p=input_dropout_rate))
        if init_weights:
            linear_max = (odim * input_size / 4) ** -0.5
            self.input_proj.state_dict()['0.weight'] = paddle.nn.initializer.Uniform(low=-linear_max, high=linear_max)
            self.input_proj.state_dict()['0.bias'] = paddle.nn.initializer.Uniform(low=-linear_max, high=linear_max)

        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: paddle.Tensor,
            x_mask: paddle.Tensor,
            offset: int = 0
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.pw_conv(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.act2(x)
        b, c, t, f = x.shape
        x = x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f])
        x, pos_emb = self.pos_enc(x, offset)
        x = self.input_proj(x)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]
