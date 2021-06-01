from paddle import nn
from model_utils.utils import brelu, make_non_pad_mask
from paddle.nn import functional as F

__all__ = ['ConvStack']


class ConvBn(nn.Layer):
    """带BN层的卷积

    :param num_channels_in: 输入通道的大小
    :type num_channels_in: int
    :param num_channels_out: 输出通道的大小
    :type num_channels_out: int
    :param kernel_size: 卷积核的大小
    :type kernel_size: int|tuple|list
    :param stride: 卷积核滑动的步数
    :type stride: int|tuple|list
    :param padding: 填充的大小
    :type padding: int|tuple|list
    :param act: 激活函数的类型, relu|brelu
    :type act: string

    :return: 带BN层的卷积
    :rtype: nn.Layer

    """

    def __init__(self, num_channels_in, num_channels_out, kernel_size, stride, padding, act):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2D(num_channels_in,
                              num_channels_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              weight_attr=None,
                              bias_attr=False,
                              data_format='NCHW')

        self.bn = nn.BatchNorm2D(num_channels_out,
                                 weight_attr=None,
                                 bias_attr=None,
                                 data_format='NCHW')
        self.act = F.relu if act == 'relu' else brelu

    def forward(self, x, x_len):
        """
        x(Tensor): audio, shape [B, C, D, T]
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        x_len = (x_len - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        # 将填充部分重置为0
        masks = make_non_pad_mask(x_len)  # [B, T]
        masks = masks.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        masks = masks.astype(x.dtype)
        x = x.multiply(masks)
        return x, x_len


class ConvStack(nn.Layer):
    """具有堆叠卷积层的卷积组

    :param feat_size: 输入音频的特征大小
    :type feat_size: int
    :param num_stacks: 堆叠卷积层的数量
    :type num_stacks: int
    """

    def __init__(self, feat_size, num_stacks):
        super().__init__()
        self.feat_size = feat_size  # D
        self.num_stacks = num_stacks

        self.conv_in = ConvBn(num_channels_in=1,
                              num_channels_out=32,
                              kernel_size=(41, 11),  # [D, T]
                              stride=(2, 3),
                              padding=(20, 5),
                              act='brelu')

        out_channel = 32
        convs = [
            ConvBn(num_channels_in=32,
                   num_channels_out=out_channel,
                   kernel_size=(21, 11),
                   stride=(2, 1),
                   padding=(10, 5),
                   act='brelu') for i in range(self.num_stacks - 1)
        ]
        self.conv_stack = nn.LayerList(convs)

        # 卷积层输出的特征大小
        output_height = (self.feat_size - 1) // 2 + 1
        for i in range(self.num_stacks - 1):
            output_height = (output_height - 1) // 2 + 1
        self.output_height = out_channel * output_height

    def forward(self, x, x_len):
        """
        x: shape [B, C, D, T]
        x_len : shape [B]
        """
        x, x_len = self.conv_in(x, x_len)
        for i, conv in enumerate(self.conv_stack):
            x, x_len = conv(x, x_len)
        return x, x_len
