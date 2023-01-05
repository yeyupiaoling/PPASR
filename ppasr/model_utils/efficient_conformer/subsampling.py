from typing import Tuple, Union

import paddle

from ppasr.model_utils.conformer.subsampling import BaseSubsampling
from ppasr.model_utils.utils.base import Conv2D, Linear


class Conv2dSubsampling2(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: paddle.nn.Layer):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = paddle.nn.Sequential(Conv2D(1, odim, 3, 2),
                                         paddle.nn.ReLU())
        self.out = paddle.nn.Sequential(Linear(odim * ((idim - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 2
        # 2 = (3 - 1) * 1
        self.right_context = 2

    def forward(
            self,
            x: paddle.Tensor,
            x_mask: paddle.Tensor,
            offset: Union[int, paddle.Tensor] = 0
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Subsample x.

        Args:
            x (paddle.Tensor): Input tensor (#batch, time, idim).
            x_mask (paddle.Tensor): Input mask (#batch, 1, time).

        Returns:
            paddle.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            paddle.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.
            paddle.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2]
