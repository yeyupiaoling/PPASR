import paddle
from paddle import nn

__all__ = ['ConvStack']


class MaskConv(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, lengths):
        """
        :param x: 卷积输入，shape[B, C, D, T]
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
        masks = masks.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        x = x.multiply(masks)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(in_channels=channel,
                               out_channels=channel // reduction,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               weight_attr=paddle.ParamAttr(),
                               bias_attr=paddle.ParamAttr())
        self.conv2 = nn.Conv2D(in_channels=channel // reduction,
                               out_channels=channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               weight_attr=paddle.ParamAttr(),
                               bias_attr=paddle.ParamAttr())
        self.act1 = nn.ReLU()
        self.act2 = nn.Hardsigmoid()

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.act1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.act2(outputs)
        return paddle.multiply(x=inputs, y=outputs)


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
    :param use_se: 是否使用SE模块
    :type use_se: bool
    :return: 带BN层的卷积
    :rtype: nn.Layer
    """

    def __init__(self, num_channels_in, num_channels_out, kernel_size, stride, padding, use_se=False):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        self.mask = MaskConv()

        self.conv = nn.Conv2D(num_channels_in,
                              num_channels_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              data_format='NCHW')

        self.bn = nn.BatchNorm2D(num_channels_out, data_format='NCHW')
        self.act = nn.Hardtanh(min=0.0, max=24.0)
        if self.use_se:
            self.se = SEModule(num_channels_out)

    def forward(self, x, x_len):
        """
        x(Tensor): audio, shape [B, C, D, T]
        """
        x = self.conv(x)
        x = self.bn(x)
        if self.use_se:
            x = self.se(x)
        x = self.act(x)

        x_len = (x_len - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        # 将填充部分重置为0
        x = self.mask(x, x_len)
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
        out_channel = 32

        self.conv_in = ConvBn(num_channels_in=1,
                              num_channels_out=32,
                              kernel_size=(41, 11),  # [D, T]
                              stride=(2, 3),
                              padding=(20, 5))

        conv_stacks = []
        for _ in range(self.num_stacks - 1):
            conv_stacks.append(ConvBn(num_channels_in=32,
                                      num_channels_out=out_channel,
                                      kernel_size=(21, 11),
                                      stride=(2, 1),
                                      padding=(10, 5),
                                      use_se=True))
        self.conv_stack = nn.LayerList(conv_stacks)

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