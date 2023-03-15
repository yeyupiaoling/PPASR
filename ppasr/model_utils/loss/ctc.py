import paddle
import paddle.nn.functional as F
from typeguard import check_argument_types


class CTCLoss(paddle.nn.Layer):
    """CTC module"""

    def __init__(
            self,
            odim: int,
            encoder_output_size: int,
            dropout_rate: float = 0.0,
            reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = paddle.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = paddle.nn.CTCLoss(reduction=reduction_type)

    def forward(self, hs_pad: paddle.Tensor, hlens: paddle.Tensor,
                ys_pad: paddle.Tensor, ys_lens: paddle.Tensor) -> paddle.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose([1, 0, 2])
        ys_pad = ys_pad.astype(paddle.int32)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.shape[1]
        return loss

    def log_softmax(self, hs_pad: paddle.Tensor) -> paddle.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            paddle.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), axis=2)

    def softmax(self, hs_pad: paddle.Tensor) -> paddle.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            paddle.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.softmax(self.ctc_lo(hs_pad), axis=2)

    def argmax(self, hs_pad: paddle.Tensor) -> paddle.Tensor:
        """argmax of frame activations

        Args:
            paddle.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            paddle.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return paddle.argmax(self.ctc_lo(hs_pad), axis=2)
