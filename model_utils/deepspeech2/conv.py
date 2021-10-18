from paddle import nn

__all__ = ['ConvStack']


class ConvStack(nn.Layer):
    """具有堆叠卷积层的卷积组

    :param feat_size: 输入音频的特征大小
    :type feat_size: int
    :param output_dim: 卷积层输出大小
    :type output_dim: int
    """

    def __init__(self, feat_size, output_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2D(1, output_dim, 3, 2),
                                  nn.ReLU(),
                                  nn.Conv2D(output_dim, output_dim, 3, 2),
                                  nn.ReLU(), )
        self.output_dim = ((feat_size - 1) // 2 - 1) // 2 * output_dim

    def forward(self, x, x_len):
        """
        x: shape [B, D, T]
        x_len : shape [B]
        """
        # [B, D, T] -> [B, T, D]
        x = x.transpose([0, 2, 1])
        # [B, T, D] -> [B, C=1, T, D]
        x = x.unsqueeze(1)
        x = self.conv(x)
        # 将数据从卷积特征映射转换为向量序列
        x = x.transpose([0, 2, 1, 3])  # [B, T, C, D]
        x = x.reshape([0, 0, -1])  # [B, T, C*D]
        x_len = ((x_len - 1) // 2 - 1) // 2
        return x, x_len
