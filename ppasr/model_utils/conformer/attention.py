import math
from typing import Tuple

import paddle
from paddle import nn
from paddle.nn import initializer as I
from ppasr.model_utils.utils.common import masked_fill
from ppasr.model_utils.utils.base import Linear

__all__ = ["MultiHeadedAttention", "RelPositionMultiHeadedAttention"]


class MultiHeadedAttention(nn.Layer):
    """Multi-Head Attention layer."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object.
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
        """
        super().__init__()
        assert n_feat % n_head == 0
        self.n_feat = n_feat
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = Linear(n_feat, n_feat)
        self.linear_k = Linear(n_feat, n_feat)
        self.linear_v = Linear(n_feat, n_feat)
        self.linear_out = Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (paddle.Tensor): Input tensor (batch, head, time1, time1).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            paddle.Tensor: Output tensor. (batch, head, time1, time1)
        """
        zero_pad = paddle.zeros([x.shape[0], x.shape[1], x.shape[2], 1], dtype=x.dtype)
        x_padded = paddle.concat([zero_pad, x], axis=-1)

        x_padded = x_padded.reshape([x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2]])
        x = x_padded[:, :, 1:].reshape(paddle.shape(x))  # [B, H, T1, T1]

        if zero_triu:
            ones = paddle.ones((x.shape[2], x.shape[3]))
            x = x * paddle.tril(ones, x.shape[3] - x.shape[2])[None, None, :, :]

        return x

    def forward_qkv(self,
                    query: paddle.Tensor,
                    key: paddle.Tensor,
                    value: paddle.Tensor
                    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Transform query, key and value.
        Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
        Returns:
            paddle.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            paddle.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            paddle.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).
        """
        n_batch = query.shape[0]

        q = self.linear_q(query).reshape([n_batch, -1, self.h, self.d_k])
        k = self.linear_k(key).reshape([n_batch, -1, self.h, self.d_k])
        v = self.linear_v(value).reshape([n_batch, -1, self.h, self.d_k])

        q = q.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)
        k = k.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)
        v = v.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
            self,
            value: paddle.Tensor,
            scores: paddle.Tensor,
            mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool)
    ) -> paddle.Tensor:
        """Compute attention context vector.
        Args:
            value (paddle.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (paddle.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (paddle.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
        Returns:
            paddle.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.shape[0]

        # When `if mask.size(2) > 0` be True:
        # 1. training.
        # 2. oonx(16/4, chunk_size/history_size), feed real cache and real mask for the 1st chunk.
        # When will `if mask.size(2) > 0` be False?
        # 1. onnx(16/-1, -1/-1, 16/0)
        # 2. jit (16/-1, -1/-1, 16/0, 16/4)
        if mask.shape[2] > 0:  # time2 > 0
            mask = mask.unsqueeze(1).equal(0)  # (batch, 1, *, time2)
            # for last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.shape[-1]]
            scores = masked_fill(scores, mask, -float('inf'))
            attn = paddle.nn.functional.softmax(scores, axis=-1)
            attn = masked_fill(attn, mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = paddle.nn.functional.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = paddle.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose([0, 2, 1, 3]).reshape([n_batch, -1, self.h * self.d_k])  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
                pos_emb: paddle.Tensor = paddle.empty([0]),
                cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
                ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Compute scaled dot product attention.
       Args:
            query (paddle.Tensor): Query tensor (#batch, time1, size).
            key (paddle.Tensor): Key tensor (#batch, time2, size).
            value (paddle.Tensor): Value tensor (#batch, time2, size).
            mask (paddle.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (paddle.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            paddle.Tensor: Output tensor (#batch, time1, d_model).
            paddle.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        if cache.shape[0] > 0:
            # last dim `d_k * 2` for (key, val)
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = paddle.concat((k, v), axis=-1)

        # scores = paddle.matmul(q,
        #    k.transpose([0, 1, 3, 2])) / math.sqrt(self.d_k)
        scores = paddle.matmul(q, k, transpose_y=True) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding."""

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object.
        Paper: https://arxiv.org/abs/1901.02860
        Args:
            n_head (int): The number of heads.
            n_feat (int): The number of features.
            dropout_rate (float): Dropout rate.
        """
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = Linear(n_feat, n_feat, bias_attr=False)
        pos_bias_u = self.create_parameter([self.h, self.d_k], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_u', pos_bias_u)
        pos_bias_v = self.create_parameter([self.h, self.d_k], default_initializer=I.XavierUniform())
        self.add_parameter('pos_bias_v', pos_bias_v)

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
                pos_emb: paddle.Tensor = paddle.empty([0]),
                cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
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
        q, k, v = self.forward_qkv(query, key, value)
        if cache.shape[0] > 0:
            # last dim `d_k * 2` for (key, val)
            key_cache, value_cache = paddle.split(cache, 2, axis=-1)
            k = paddle.concat([key_cache, k], axis=2)
            v = paddle.concat([value_cache, v], axis=2)
        # We do cache slicing in encoder.forward_chunk, since it's
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
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache
