from typing import List
from typing import Optional, Tuple

import paddle
from paddle import nn
from typeguard import check_argument_types

from ppasr.model_utils.conformer.attention import MultiHeadedAttention
from ppasr.model_utils.utils.base import Embedding, LayerNorm, Linear
from ppasr.model_utils.conformer.embedding import PositionalEncoding
from ppasr.model_utils.conformer.positionwise import PositionwiseFeedForward
from ppasr.model_utils.utils.mask import (subsequent_mask, make_non_pad_mask)


class BiTransformerDecoder(nn.Layer):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 r_num_blocks: int = 0,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 input_layer: str = "embed",
                 use_output_layer: bool = True,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 max_len: int = 5000):
        assert check_argument_types()

        nn.Layer.__init__(self)
        self.left_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after,
            max_len)

        self.right_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after,
            max_len)

    def forward(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            ys_in_pad: paddle.Tensor,
            ys_in_lens: paddle.Tensor,
            r_ys_in_pad: paddle.Tensor,
            reverse_weight: float = 0.0,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad, ys_in_lens)
        r_x = paddle.zeros([1])
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad, ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            tgt: paddle.Tensor,
            tgt_mask: paddle.Tensor,
            cache: Optional[List[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out, maxlen_out)
                      dtype=paddle.bool
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt_mask, cache)


class TransformerDecoder(nn.Layer):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type, `embed`
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding module
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 input_layer: str = "embed",
                 use_output_layer: bool = True,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 max_len: int = 5000):

        assert check_argument_types()

        nn.Layer.__init__(self)
        self.selfattention_layer_type = 'selfattn'
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = nn.Sequential(
                Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate, max_len=max_len), )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = LayerNorm(attention_dim, epsilon=1e-12)
        self.use_output_layer = use_output_layer
        self.output_layer = Linear(attention_dim, vocab_size)

        self.decoders = nn.LayerList([
            DecoderLayer(
                size=attention_dim,
                self_attn=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate),
                src_attn=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate),
                feed_forward=PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after, ) for _ in range(num_blocks)
        ])

    def forward(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            ys_in_pad: paddle.Tensor,
            ys_in_lens: paddle.Tensor,
            r_ys_in_pad: paddle.Tensor = paddle.empty([0]),
            reverse_weight: float = 0.0) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (make_non_pad_mask(ys_in_lens).unsqueeze(1))
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.shape[-1]).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, paddle.to_tensor(0.0), olens

    def forward_one_step(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            tgt: paddle.Tensor,
            tgt_mask: paddle.Tensor,
            cache: Optional[List[paddle.Tensor]] = None,
    ) -> Tuple[paddle.Tensor, List[paddle.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out, maxlen_out)
                      dtype=paddle.bool
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
                y.shape` is (batch, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = paddle.nn.functional.log_softmax(self.output_layer(y), axis=-1)
        return y, new_cache


class DecoderLayer(nn.Layer):
    """Single decoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (nn.Layer): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (nn.Layer): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (nn.Layer): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            src_attn: nn.Layer,
            feed_forward: nn.Layer,
            dropout_rate: float,
            normalize_before: bool = True,
            concat_after: bool = False, ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, epsilon=1e-12)
        self.norm2 = LayerNorm(size, epsilon=1e-12)
        self.norm3 = LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = Linear(size + size, size)
            self.concat_linear2 = Linear(size + size, size)
        else:
            self.concat_linear1 = nn.Identity()
            self.concat_linear2 = nn.Identity()

    def forward(
            self,
            tgt: paddle.Tensor,
            tgt_mask: paddle.Tensor,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            cache: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute decoded features.
        Args:
            tgt (paddle.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (paddle.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (paddle.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (paddle.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (paddle.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).
        Returns:
            paddle.Tensor: Output tensor (#batch, maxlen_out, size).
            paddle.Tensor: Mask for output tensor (#batch, maxlen_out).
            paddle.Tensor: Encoded memory (#batch, maxlen_in, size).
            paddle.Tensor: Encoded memory mask (#batch, maxlen_in).
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == [
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ], f"{cache.shape} == {[tgt.shape[0], tgt.shape[1] - 1, self.size]}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = paddle.concat((tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0]), axis=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = paddle.concat((x, self.src_attn(x, memory, memory, memory_mask)[0]), axis=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask)[0])
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = paddle.concat([cache, x], axis=1)

        return x, tgt_mask, memory, memory_mask
