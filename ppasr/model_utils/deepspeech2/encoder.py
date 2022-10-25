import paddle
from paddle import nn

from ppasr.model_utils.deepspeech2.conv import Conv2dSubsampling4Pure


class CRNNEncoder(nn.Layer):
    def __init__(self,
                 input_dim,
                 vocab_size,
                 global_cmvn=None,
                 num_rnn_layers=4,
                 rnn_size=1024,
                 rnn_direction='forward',
                 use_gru=False):
        super().__init__()
        self.rnn_size = rnn_size
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.num_rnn_layers = num_rnn_layers
        self.rnn_direction = rnn_direction
        self.use_gru = use_gru
        self.conv = Conv2dSubsampling4Pure(input_dim, 32)

        self.global_cmvn = global_cmvn
        self.output_dim = self.conv.output_dim

        i_size = self.conv.output_dim
        self.rnn = nn.LayerList()
        self.layernorm_list = nn.LayerList()
        if rnn_direction == 'bidirect' or rnn_direction == 'bidirectional':
            layernorm_size = 2 * rnn_size
        elif rnn_direction == 'forward':
            layernorm_size = rnn_size
        else:
            raise Exception("Wrong rnn direction")
        for i in range(0, num_rnn_layers):
            if i == 0:
                rnn_input_size = i_size
            else:
                rnn_input_size = layernorm_size
            if use_gru is True:
                self.rnn.append(
                    nn.GRU(input_size=rnn_input_size,
                           hidden_size=rnn_size,
                           num_layers=1,
                           direction=rnn_direction))
            else:
                self.rnn.append(
                    nn.LSTM(input_size=rnn_input_size,
                            hidden_size=rnn_size,
                            num_layers=1,
                            direction=rnn_direction))
            self.layernorm_list.append(nn.LayerNorm(layernorm_size))
            self.output_dim = layernorm_size

    @property
    def output_size(self):
        return self.output_dim

    def forward(self, x, x_lens, init_state_h_box=None, init_state_c_box=None):
        """Compute Encoder outputs

        Args:
            x (Tensor): [B, T, D]
            x_lens (Tensor): [B]
            init_state_h_box(Tensor): init_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
            init_state_c_box(Tensor): init_states c for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        Return:
            x (Tensor): encoder outputs, [B, T, D]
            x_lens (Tensor): encoder length, [B]
            final_state_h_box(Tensor): final_states h for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
            final_state_c_box(Tensor): final_states c for RNN layers: [num_rnn_layers * num_directions, batch_size, hidden_size]
        """
        if init_state_h_box is not None:
            if self.use_gru is True:
                init_state_h_list = paddle.split(init_state_h_box, self.num_rnn_layers, axis=0)
                init_state_list = init_state_h_list
            else:
                init_state_h_list = paddle.split(init_state_h_box, self.num_rnn_layers, axis=0)
                init_state_c_list = paddle.split(init_state_c_box, self.num_rnn_layers, axis=0)
                init_state_list = [(init_state_h_list[i], init_state_c_list[i]) for i in range(self.num_rnn_layers)]
        else:
            init_state_list = [None] * self.num_rnn_layers

        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        x, x_lens = self.conv(x, x_lens)
        final_chunk_state_list = []
        for i in range(0, self.num_rnn_layers):
            x, final_state = self.rnn[i](x, init_state_list[i], x_lens)  # [B, T, D]
            final_chunk_state_list.append(final_state)
            x = self.layernorm_list[i](x)

        if self.use_gru is True:
            final_chunk_state_h_box = paddle.concat(final_chunk_state_list, axis=0)
            final_chunk_state_c_box = init_state_c_box
        else:
            final_chunk_state_h_list = [final_chunk_state_list[i][0] for i in range(self.num_rnn_layers)]
            final_chunk_state_c_list = [final_chunk_state_list[i][1] for i in range(self.num_rnn_layers)]
            final_chunk_state_h_box = paddle.concat(final_chunk_state_h_list, axis=0)
            final_chunk_state_c_box = paddle.concat(final_chunk_state_c_list, axis=0)

        return x, x_lens, final_chunk_state_h_box, final_chunk_state_c_box
