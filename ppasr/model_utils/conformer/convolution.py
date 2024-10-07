from typing import Tuple

import paddle
from paddle import nn
from typeguard import typechecked

from ppasr.model_utils.utils.common import masked_fill
from ppasr.model_utils.utils.base import Conv1D

__all__ = ['ConvolutionModule']


class ConvolutionModule(nn.Layer):
    """ConvolutionModule in Conformer model."""

    @typechecked
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Layer = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            activation (nn.Layer): Activation Layer.
            norm (str): Normalization type, 'batch_norm' or 'layer_norm'
            causal (bool): Whether use causal convolution or not
            bias (bool): Whether Conv with bias or not
        """
        super().__init__()
        self.pointwise_conv1 = Conv1D(channels,
                                      2 * channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias_attr=None
                                      if bias else False)

        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0:
        #    it's a causal convolution, the input will be padded with
        #    `self.lorder` frames on the left in forward (causal conv impl).
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = Conv1D(channels,
                                     channels,
                                     kernel_size,
                                     stride=1,
                                     padding=padding,
                                     groups=channels,
                                     bias_attr=None
                                     if bias else False)

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1D(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = Conv1D(channels,
                                      channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias_attr=None
                                      if bias else False)
        self.activation = activation

    def forward(
            self,
            x: paddle.Tensor,
            mask_pad: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
            cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute convolution module.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, channels).
            mask_pad (paddle.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (paddle.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            paddle.Tensor: Output tensor (#batch, time, channels).
            paddle.Tensor: Output cache tensor (#batch, channels, time')
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose([0, 2, 1])  # [B, C, T]

        # mask batch padding
        if mask_pad.shape[2] > 0:  # time > 0
            # TODO 需要检查这个
            x = masked_fill(x, mask_pad, 0.0)

        if self.lorder > 0:
            if cache.shape[2] == 0:  # cache_t == 0
                x = nn.functional.pad(x, [self.lorder, 0], 'constant', 0.0, data_format='NCL')
            else:
                assert cache.shape[0] == x.shape[0]  # B
                assert cache.shape[1] == x.shape[1]  # C
                x = paddle.concat((cache, x), axis=2)

            assert (x.shape[2] > self.lorder)
            new_cache = x[:, :, -self.lorder:]  # [B, C, T]
        else:
            # It's better we just return None if no cache is requried,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = paddle.zeros([0, 0, 0], dtype=x.dtype)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, axis=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, T, C]
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, C, T]
        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad.shape[2] > 0:  # time > 0
            # TODO 需要检查这个
            x = masked_fill(x, mask_pad, 0.0)

        x = x.transpose([0, 2, 1])  # [B, T, C]
        return x, new_cache
