import paddle
from paddle import nn

__all__ = ['RNNStack']


class RNNStack(nn.Layer):
    """RNN组与堆叠双向简单RNN或GRU层

    :param i_size: GRU层的输入大小
    :type i_size: int
    :param h_size: GRU层的隐层大小
    :type h_size: int
    :param num_rnn_layers: rnn层数
    :type num_rnn_layers: int

    :return: RNN组的输出层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, num_rnn_layers: int):
        super().__init__()
        self.rnn = nn.LayerList()
        self.norm_list = nn.LayerList()
        self.output_dim = h_size
        self.num_rnn_layers = num_rnn_layers
        for i in range(0, self.num_rnn_layers):
            if i == 0:
                rnn_input_size = i_size
            else:
                rnn_input_size = h_size
            self.rnn.append(nn.GRU(input_size=rnn_input_size,
                                   hidden_size=h_size,
                                   num_layers=1,
                                   direction="forward"))
            self.norm_list.append(nn.LayerNorm(h_size))

    def forward(self, x, x_lens, init_state_h_box=None):
        if init_state_h_box is not None:
            init_state_list = paddle.split(init_state_h_box, self.num_rnn_layers, axis=0)
        else:
            init_state_list = [None] * self.num_rnn_layers

        final_chunk_state_list = []
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, init_state_list[i], x_lens)  # [B, T, D]
            final_chunk_state_list.append(final_state)
            x = self.norm_list[i](x)

        final_chunk_state_h_box = paddle.concat(final_chunk_state_list, axis=0)
        return x, x_lens, final_chunk_state_h_box
