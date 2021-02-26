import paddle
import paddle.nn as nn
from paddle.nn.initializer import KaimingNormal


class GLU(nn.Layer):
    def __init__(self):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        a, b = paddle.split(x, num_or_sections=2, axis=1)
        act_b = self.sigmoid(b)
        out = paddle.multiply(x=a, y=act_b)
        return out


class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, p=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1D(in_channels, out_channels, kernel_size, stride, padding, weight_attr=KaimingNormal())
        self.conv = nn.utils.weight_norm(self.conv)
        self.act = nn.Hardtanh()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class PPASR(nn.Layer):
    def __init__(self, vocabulary):
        super(PPASR, self).__init__()
        self.output_units = len(vocabulary) + 1
        self.conv1 = ConvBlock(161, 500, 48, 2, padding=97, p=0.2)
        self.conv2 = ConvBlock(500, 500, 7, 1, p=0.3)
        self.conv3 = ConvBlock(500, 1000, 32, 1, p=0.3)
        self.conv4 = ConvBlock(1000, 2000, 1, 1, p=0.3)
        self.out = nn.utils.weight_norm(nn.Conv1D(2000, self.output_units, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        for i in range(7):
            x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)
        return x
