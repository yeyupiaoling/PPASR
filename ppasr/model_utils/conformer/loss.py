import paddle
import paddle.nn.functional as F
from paddle import nn
from typeguard import check_argument_types

from ppasr.model_utils.utils.common import masked_fill


class LabelSmoothingLoss(nn.Layer):
    """Label-smoothing loss.
    In a standard CE loss, the label's data distribution is:
        [0,1,2] ->
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.
        e.g.
        smoothing=0.1
        [0,1,2] ->
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ]

    """

    def __init__(self,
                 size: int,
                 padding_idx: int,
                 smoothing: float,
                 normalize_length: bool = False):
        """Label-smoothing loss.

        Args:
            size (int): the number of class
            padding_idx (int): padding class id which will be ignored for loss
            smoothing (float): smoothing rate (0.0 means the conventional CE)
            normalize_length (bool):
                True, normalize loss by sequence length;
                False, normalize loss by batch size.
                Defaults to False.
        """
        super().__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.normalize_length = normalize_length
        self.criterion = nn.KLDivLoss(reduction="none")

    def forward(self, x: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        """Compute loss between x and target.
        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (paddle.Tensor): prediction (batch, seqlen, class)
            target (paddle.Tensor):
                target signal masked with self.padding_id (batch, seqlen)
        Returns:
            loss (paddle.Tensor) : The KL loss, scalar float value
        """
        B, T, D = x.shape
        assert D == self.size
        x = x.reshape((-1, self.size))
        target = target.reshape([-1])

        # use zeros_like instead of torch.no_grad() for true_dist,
        # since no_grad() can not be exported by JIT
        true_dist = paddle.full_like(x, self.smoothing / (self.size - 1))
        ignore = target == self.padding_idx  # (B,)

        target = masked_fill(target, ignore, 0)  # avoid -1 index
        # true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        target_mask = F.one_hot(target, self.size)
        true_dist *= (1 - target_mask)
        true_dist += target_mask * self.confidence

        kl = self.criterion(F.log_softmax(x, axis=1), true_dist)

        # total = len(target) - int(ignore.sum())
        total = len(target) - int(ignore.astype(target.dtype).sum())
        denom = total if self.normalize_length else B
        # numer = (kl * (1 - ignore)).sum()
        numer = masked_fill(kl, ignore.unsqueeze(1), 0).sum()
        return numer / denom


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
        loss = loss / ys_hat.shape[0]
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
