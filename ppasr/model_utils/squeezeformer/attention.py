import math
from typing import Tuple

import paddle
from paddle.nn import initializer as I

from ppasr.model_utils.utils.base import Linear
from ppasr.model_utils.conformer.attention import MultiHeadedAttention

__all__ = ['RelPositionMultiHeadedAttention']


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, do_rel_shift=False, adaptive_scale=False, init_weights=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = Linear(n_feat, n_feat)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.do_rel_shift = do_rel_shift
        pos_bias_u = self.create_parameter([self.h, self.d_k], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_u', pos_bias_u)
        pos_bias_v = self.create_parameter([self.h, self.d_k], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_v', pos_bias_v)
        self.adaptive_scale = adaptive_scale
        ada_scale = self.create_parameter([1, 1, n_feat], default_initializer=I.Constant(1.0))
        self.add_parameter('ada_scale', ada_scale)
        ada_bias = self.create_parameter([1, 1, n_feat], default_initializer=I.Constant(0.0))
        self.add_parameter('ada_bias', ada_bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        input_max = (self.h * self.d_k) ** -0.5
        self.linear_q._param_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_q._bias_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_k._param_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_k._bias_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_v._param_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_v._bias_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_pos._param_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_pos._bias_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_out._param_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)
        self.linear_out._bias_attr = paddle.nn.initializer.Uniform(low=-input_max, high=input_max)

    def forward(self, query: paddle.Tensor,
                key: paddle.Tensor, value: paddle.Tensor,
                mask: paddle.Tensor = paddle.ones((0, 0, 0), dtype=paddle.bool),
                pos_emb: paddle.Tensor = paddle.empty([0]),
                cache: paddle.Tensor = paddle.zeros((0, 0, 0, 0))
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
            mask (paddle.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (paddle.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (paddle.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            paddle.Tensor: Output tensor (#batch, time1, d_model).
            paddle.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        if self.adaptive_scale:
            query = self.ada_scale * query + self.ada_bias
            key = self.ada_scale * key + self.ada_bias
            value = self.ada_scale * value + self.ada_bias

        q, k, v = self.forward_qkv(query, key, value)
        if cache.shape[0] > 0:
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = paddle.concat((k, v), axis=-1)

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).reshape([n_batch_pos, -1, self.h, self.d_k])
        p = p.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        # q_with_bias_u = (q + self.pos_bias_u).transpose([0, 2, 1, 3])
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)
        # (batch, head, time1, d_k)
        # q_with_bias_v = (q + self.pos_bias_v).transpose([0, 2, 1, 3])
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)

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
        if self.do_rel_shift:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache
