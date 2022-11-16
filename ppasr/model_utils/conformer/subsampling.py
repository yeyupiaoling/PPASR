from typing import Tuple

import paddle
from paddle import nn
from ppasr.model_utils.utils.base import Conv2D, Linear

__all__ = [
    "LinearNoSubsampling", "Conv2dSubsampling4", "Conv2dSubsampling6",
    "Conv2dSubsampling8"
]


class BaseSubsampling(nn.Layer):
    def __init__(self):
        super().__init__()
        # window size = (1 + right_context) + (chunk_size -1) * subsampling_rate
        self.right_context = 0
        # stride = subsampling_rate * chunk_size
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer):
        """Construct an linear object.
        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc_class (PositionalEncoding): position encoding class
        """
        super().__init__()
        self.out = nn.Sequential(nn.Linear(idim, odim),
                                 nn.LayerNorm(odim, epsilon=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.ReLU(), )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int = 0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Input x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            paddle.Tensor: positional encoding
            paddle.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer):
        """Construct an Conv2dSubsampling4 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(Conv2D(1, odim, 3, 2),
                                  nn.ReLU(),
                                  Conv2D(odim, odim, 3, 2),
                                  nn.ReLU(), )
        self.out = nn.Sequential(nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 4
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int = 0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            paddle.Tensor: positional encoding
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer):
        """Construct an Conv2dSubsampling6 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
            pos_enc (PositionalEncoding): Custom position encoding layer.
        """
        super().__init__()
        self.conv = nn.Sequential(Conv2D(1, odim, 3, 2),
                                  nn.ReLU(),
                                  Conv2D(odim, odim, 5, 3),
                                  nn.ReLU(), )
        self.pos_enc = pos_enc_class
        # O = (I - F + Pstart + Pend) // S + 1
        # when Padding == 0, O = (I - F - S) // S
        self.linear = Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int = 0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            paddle.Tensor: positional encoding
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer):
        """Construct an Conv2dSubsampling8 object.

        Args:
            idim (int): Input dimension.
            odim (int): Output dimension.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(Conv2D(1, odim, 3, 2),
                                  nn.ReLU(),
                                  Conv2D(odim, odim, 3, 2),
                                  nn.ReLU(),
                                  Conv2D(odim, odim, 3, 2),
                                  nn.ReLU(), )
        self.linear = Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int = 0
                ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).
            offset (int): position encoding offset.
        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            paddle.Tensor: positional encoding
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([b, -1, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]
