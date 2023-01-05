import paddle

from paddle.nn import initializer as I

from ppasr.model_utils.utils.base import Linear

__all__ = ["PositionwiseFeedForward"]


class PositionwiseFeedForward(paddle.nn.Layer):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (paddle.nn.Layer): Activation function
    """

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: paddle.nn.Layer = paddle.nn.ReLU(),
                 adaptive_scale: bool = False,
                 init_weights: bool = False):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.idim = idim
        self.hidden_units = hidden_units
        self.w_1 = Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = paddle.nn.Dropout(dropout_rate)
        self.w_2 = Linear(hidden_units, idim)
        self.adaptive_scale = adaptive_scale
        ada_scale = self.create_parameter([1, 1, idim], default_initializer=I.XavierUniform())
        self.add_parameter('ada_scale', ada_scale)
        ada_bias = self.create_parameter([1, 1, idim], default_initializer=I.XavierUniform())
        self.add_parameter('ada_bias', ada_bias)

        if init_weights:
            self.init_weights()

    def init_weights(self):
        ffn1_max = self.idim ** -0.5
        ffn2_max = self.hidden_units ** -0.5
        self.w_1._param_attr = paddle.nn.initializer.Uniform(low=-ffn1_max, high=ffn1_max)
        self.w_1._bias_attr = paddle.nn.initializer.Uniform(low=-ffn1_max, high=ffn1_max)
        self.w_2._param_attr = paddle.nn.initializer.Uniform(low=-ffn2_max, high=ffn2_max)
        self.w_2._bias_attr = paddle.nn.initializer.Uniform(low=-ffn2_max, high=ffn2_max)

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        if self.adaptive_scale:
            xs = self.ada_scale * xs + self.ada_bias
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
