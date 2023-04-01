import math
from typing import Tuple, Optional

import paddle
from paddle.nn import initializer as I
import paddle.nn.functional as F

from ppasr.model_utils.utils.base import Linear
from ppasr.model_utils.conformer.attention import MultiHeadedAttention

__all__ = ['GroupedRelPositionMultiHeadedAttention']

from ppasr.model_utils.utils.common import masked_fill


class GroupedRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper:
        https://arxiv.org/abs/1901.02860
        https://arxiv.org/abs/2109.01163
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, group_size=3):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = Linear(n_feat, n_feat)
        self.group_size = group_size
        self.d_k = n_feat // n_head  # for GroupedAttention
        self.n_feat = n_feat
        pos_bias_u = self.create_parameter([self.h, self.d_k * self.group_size], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_u', pos_bias_u)
        pos_bias_v = self.create_parameter([self.h, self.d_k * self.group_size], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_v', pos_bias_v)

    def pad4group(self, Q, K, V, P, mask, group_size: int = 3):
        """
        q: (#batch, time1, size) -> (#batch, head, time1, size/head)
        k,v: (#batch, time2, size) -> (#batch, head, time2, size/head)
        p: (#batch, time2, size)
        """
        # Compute Overflows
        overflow_Q = Q.shape[2] % group_size
        overflow_KV = K.shape[2] % group_size

        padding_Q = paddle.full([4], 0, dtype="int32")
        padding_q = (group_size - overflow_Q) * int(overflow_Q // (overflow_Q + 1e-17))
        padding_Q[3] = padding_q
        padding_KV = paddle.full([4], 0, dtype="int32")
        padding_KV[3] = (group_size - overflow_KV) * int(overflow_KV // (overflow_KV + 1e-17))

        batch_size, _, seq_len_KV, _ = K.shape

        # Input Padding (B, T, D) -> (B, T + P, D)

        Q = F.pad(Q, padding_Q, mode='constant', value=0.0)
        K = F.pad(K, padding_KV, mode='constant', value=0.0)
        V = F.pad(V, padding_KV, mode='constant', value=0.0)

        if mask is not None and mask.shape[2] > 0:  # time2 > 0:
            mask = mask[:, ::group_size, ::group_size]

        Q = Q.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.h, self.d_k * group_size]).transpose([0, 2, 1, 3])
        K = K.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.h, self.d_k * group_size]).transpose([0, 2, 1, 3])
        V = V.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.h, self.d_k * group_size]).transpose([0, 2, 1, 3])

        # process pos_emb
        P_batch_size = P.shape[0]
        overflow_P = P.shape[1] % group_size
        padding_P = paddle.full([2], 0, dtype="int32")
        padding_P[1] = group_size - overflow_P if overflow_P else 0
        P = F.pad(P, padding_P, mode='constant', value=0.0, data_format='NLC')
        P = P.reshape([P_batch_size, -1, self.h, self.d_k * group_size]).transpose([0, 2, 1, 3])

        return Q, K, V, P, mask, padding_q

    def forward_attention(
            self, value: paddle.Tensor, scores: paddle.Tensor,
            mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
            padding_q: Optional[int] = None
    ) -> paddle.Tensor:
        """Compute attention context vector.

        Args:
            value (paddle.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (paddle.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (paddle.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            padding_q : for GroupedAttention in efficent conformer

        Returns:
            paddle.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.shape[0]
        # NOTE(xcsong): When will `if mask.shape[2] > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.shape[2] > 0:  # time2 > 0
            mask = mask.unsqueeze(1).equal(0)  # (batch, 1, *, time2)
            # for last chunk, time2 might be larger than scores.shape[-1]
            mask = mask[:, :, :, :scores.shape[-1]]
            scores = masked_fill(scores, mask, -float('inf'))
            attn = paddle.nn.functional.softmax(scores, axis=-1)
            attn = masked_fill(attn, mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = paddle.nn.functional.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = paddle.matmul(p_attn, value)  # (batch, head, time1, d_k)
        # n_feat!=h*d_k may be happened in GroupAttention
        x = x.transpose([0, 2, 1, 3]).reshape([n_batch, -1, self.n_feat])  # (batch, time1, d_model)

        if padding_q is not None:
            # for GroupedAttention in efficent conformer
            x = x[:, :x.shape[1] - padding_q]

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: paddle.Tensor,
                key: paddle.Tensor, value: paddle.Tensor,
                mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
                pos_emb: paddle.Tensor = paddle.empty([0]),
                cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        p = self.linear_pos(pos_emb)  # (#batch, time2, size)
        if cache.shape[0] > 0:
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = paddle.concat((k, v), axis=-1)

        # May be k and p does not match.  eg. time2=18+18/2=27 > mask=36/2=18
        if mask is not None and mask.shape[2] > 0:
            time2 = mask.shape[2]
            k = k[:, :, -time2:, :]
            v = v[:, :, -time2:, :]

        # q k v p: (batch, head, time1, d_k)
        q, k, v, p, mask, padding_q = self.pad4group(q, k, v, p, mask, self.group_size)

        # q_with_bias_u & q_with_bias_v = (batch, head, time1, d_k)
        q = q.transpose([0, 2, 1, 3])  # (batch, time1, head, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose([0, 2, 1, 3])
        q_with_bias_v = (q + self.pos_bias_v).transpose([0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        # matrix_ac = paddle.matmul(q_with_bias_u, k.transpose([0, 1, 3, 2]))
        matrix_ac = paddle.matmul(q_with_bias_u, k, transpose_y=True)

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        # matrix_bd = paddle.matmul(q_with_bias_v, p.transpose([0, 1, 3, 2]))
        matrix_bd = paddle.matmul(q_with_bias_v, p, transpose_y=True)
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k * self.group_size)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask, padding_q), new_cache
