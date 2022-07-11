import paddle
from paddle import nn

__all__ = ['RNNStack']


class RNNForward(nn.Layer):
    def __init__(self, rnn_input_size, h_size, use_gru):
        super().__init__()
        if use_gru:
            self.rnn = nn.GRU(input_size=rnn_input_size,
                              hidden_size=h_size,
                              direction="forward")
        else:
            self.rnn = nn.LSTM(input_size=rnn_input_size,
                               hidden_size=h_size,
                               direction="forward")
        self.norm = nn.LayerNorm(h_size)

    def forward(self, x, x_lens, init_state):
        x, final_state = self.rnn(x, init_state, x_lens)  # [B, T, D]
        x = self.norm(x)
        return x, final_state


class RNNStack(nn.Layer):
    """堆叠单向GRU层

    :param i_size: GRU层的输入大小
    :type i_size: int
    :param h_size: GRU层的隐层大小
    :type h_size: int
    :param num_rnn_layers: rnn层数
    :type num_rnn_layers: int
    :param use_gru: 使用使用GRU，否则使用LSTM
    :type use_gru: bool

    :return: RNN组的输出层
    :rtype: nn.Layer
    """

    def __init__(self, i_size: int, h_size: int, num_rnn_layers: int, use_gru: bool):
        super().__init__()
        self.rnn = nn.LayerList()
        self.output_dim = h_size
        self.use_gru = use_gru
        self.num_rnn_layers = num_rnn_layers
        self.rnn.append(RNNForward(rnn_input_size=i_size, h_size=h_size, use_gru=use_gru))
        for i in range(0, self.num_rnn_layers - 1):
            self.rnn.append(RNNForward(rnn_input_size=h_size, h_size=h_size, use_gru=use_gru))

    def forward(self, x, x_lens, init_state_h_box=None, init_state_c_box=None):
        if init_state_h_box is not None:
            if self.use_gru:
                init_state_h_list = paddle.split(init_state_h_box, self.num_rnn_layers, axis=0)
                init_state_list = init_state_h_list
            else:
                init_state_h_list = paddle.split(init_state_h_box, self.num_rnn_layers, axis=0)
                init_state_c_list = paddle.split(init_state_c_box, self.num_rnn_layers, axis=0)
                init_state_list = [(init_state_h_list[i], init_state_c_list[i]) for i in range(self.num_rnn_layers)]
        else:
            init_state_list = [None] * self.num_rnn_layers
        final_chunk_state_list = []
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, x_lens, init_state_list[i])
            final_chunk_state_list.append(final_state)

        if self.use_gru:
            final_chunk_state_h_box = paddle.concat(final_chunk_state_list, axis=0)
            final_chunk_state_c_box = init_state_c_box
        else:
            final_chunk_state_h_list = [final_chunk_state_list[i][0] for i in range(self.num_rnn_layers)]
            final_chunk_state_c_list = [final_chunk_state_list[i][1] for i in range(self.num_rnn_layers)]
            final_chunk_state_h_box = paddle.concat(final_chunk_state_h_list, axis=0)
            final_chunk_state_c_box = paddle.concat(final_chunk_state_c_list, axis=0)
        return x, final_chunk_state_h_box, final_chunk_state_c_box
