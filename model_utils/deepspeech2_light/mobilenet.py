import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import AdaptiveAvgPool2D
from paddle.nn import Conv2D, BatchNorm
from paddle.nn.functional import hardsigmoid
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay


class ConvBNLayer(nn.Layer):
    def __init__(self, num_channels, filter_size, num_filters, stride, padding, num_groups=1, act='hard_swish'):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(in_channels=num_channels,
                            out_channels=num_filters,
                            kernel_size=filter_size,
                            stride=stride,
                            padding=padding,
                            groups=num_groups,
                            weight_attr=ParamAttr(initializer=KaimingNormal()),
                            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters,
                                     act=act,
                                     param_attr=ParamAttr(regularizer=L2Decay(0.0)),
                                     bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(in_channels=channel,
                            out_channels=channel // reduction,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            weight_attr=ParamAttr(),
                            bias_attr=ParamAttr())
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs)


class DepthwiseSeparable(nn.Layer):
    def __init__(self, num_channels, num_filters1, num_filters2, num_groups, stride, scale, dw_size=3,
                 padding=1, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels,
                                           num_filters=int(num_filters1 * scale),
                                           filter_size=dw_size,
                                           stride=stride,
                                           padding=padding,
                                           num_groups=int(num_groups * scale))
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(num_channels=int(num_filters1 * scale),
                                           filter_size=1,
                                           num_filters=int(num_filters2 * scale),
                                           stride=1,
                                           padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self.use_se:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


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


class MobileNetV1(nn.Layer):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(num_channels=1,
                                 filter_size=3,
                                 num_filters=int(32 * scale),
                                 stride=2,
                                 padding=1)

        conv2_1 = DepthwiseSeparable(num_channels=int(32 * scale),
                                     num_filters1=32,
                                     num_filters2=64,
                                     num_groups=32,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(num_channels=int(64 * scale),
                                     num_filters1=64,
                                     num_filters2=128,
                                     num_groups=64,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(num_channels=int(128 * scale),
                                     num_filters1=128,
                                     num_filters2=128,
                                     num_groups=128,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(num_channels=int(128 * scale),
                                     num_filters1=128,
                                     num_filters2=256,
                                     num_groups=128,
                                     stride=(2, 1),
                                     scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(num_channels=int(256 * scale),
                                     num_filters1=256,
                                     num_filters2=256,
                                     num_groups=256,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(num_channels=int(256 * scale),
                                     num_filters1=256,
                                     num_filters2=512,
                                     num_groups=256,
                                     stride=(2, 1),
                                     scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(1):
            conv5 = DepthwiseSeparable(num_channels=int(512 * scale),
                                       num_filters1=512,
                                       num_filters2=512,
                                       num_groups=512,
                                       stride=1,
                                       dw_size=5,
                                       padding=2,
                                       scale=scale,
                                       use_se=False)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(num_channels=int(512 * scale),
                                     num_filters1=512,
                                     num_filters2=1024,
                                     num_groups=512,
                                     stride=(2, 1),
                                     dw_size=5,
                                     padding=2,
                                     scale=scale,
                                     use_se=True)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(num_channels=int(1024 * scale),
                                   num_filters1=1024,
                                   num_filters2=1024,
                                   num_groups=1024,
                                   stride=1,
                                   dw_size=5,
                                   padding=2,
                                   use_se=True,
                                   scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale * 5)
        self.mask1 = MaskConv()
        self.mask2 = MaskConv()

    def forward(self, inputs, x_len):
        x = self.conv1(inputs)
        x_len = (x_len - 3 + 2 * 1) // 2 + 1
        x = self.mask1(x, x_len)
        x = self.block_list(x)
        x = self.pool(x)
        x_len = (x_len - 2) // 2 + 1
        x = self.mask2(x, x_len)
        return x, x_len
