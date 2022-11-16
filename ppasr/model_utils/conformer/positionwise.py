import paddle
from paddle import nn

from ppasr.model_utils.utils.base import Linear


class PositionwiseFeedForward(nn.Layer):
    """Positionwise feed forward layer."""

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Layer = nn.ReLU()):
        """Construct a PositionwiseFeedForward object.

        FeedForward are appied on each position of the sequence.
        The output dim is same with the input dim.

        Args:
            idim (int): Input dimenstion.
            hidden_units (int): The number of hidden units.
            dropout_rate (float): Dropout rate.
            activation (paddle.nn.Layer): Activation function
        """
        super().__init__()
        self.w_1 = Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = Linear(hidden_units, idim)

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        """Forward function.
        Args:
            xs: input tensor (B, Lmax, D)
        Returns:
            output tensor, (B, Lmax, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
