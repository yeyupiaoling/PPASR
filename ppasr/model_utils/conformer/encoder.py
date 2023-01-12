from typing import Optional, Tuple

import paddle
from paddle import nn
from typeguard import check_argument_types

from ppasr.model_utils.conformer.attention import MultiHeadedAttention
from ppasr.model_utils.conformer.attention import RelPositionMultiHeadedAttention
from ppasr.model_utils.utils.base import LayerNorm, Linear
from ppasr.model_utils.conformer.convolution import ConvolutionModule
from ppasr.model_utils.conformer.embedding import NoPositionalEncoding
from ppasr.model_utils.conformer.embedding import PositionalEncoding
from ppasr.model_utils.conformer.embedding import RelPositionalEncoding
from ppasr.model_utils.conformer.positionwise import PositionwiseFeedForward
from ppasr.model_utils.conformer.subsampling import Conv2dSubsampling4
from ppasr.model_utils.conformer.subsampling import Conv2dSubsampling6
from ppasr.model_utils.conformer.subsampling import Conv2dSubsampling8
from ppasr.model_utils.conformer.subsampling import LinearNoSubsampling
from ppasr.model_utils.utils.common import get_activation
from ppasr.model_utils.utils.mask import add_optional_chunk_mask
from ppasr.model_utils.utils.mask import make_non_pad_mask


class ConformerEncoder(nn.Layer):
    """Conformer encoder module."""

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "rel_pos",
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: paddle.nn.Layer = None,
            use_dynamic_left_chunk: bool = False,
            macaron_style: bool = True,
            activation_type: str = "swish",
            use_cnn_module: bool = True,
            cnn_module_kernel: int = 15,
            causal: bool = False,
            cnn_module_norm: str = "layer_norm",
            max_len: int = 5000
    ):
        """Construct ConformerEncoder

        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[paddle.nn.Layer]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether you use dynamic left chunk in dynamic chunk training

            input_size to use_dynamic_chunk, see in BaseEncoder
            macaron_style (bool): Whether to use macaron style for positionwise layer.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            idim=input_size,
            odim=output_size,
            dropout_rate=dropout_rate,
            pos_enc_class=pos_enc_class(
                d_model=output_size,
                dropout_rate=positional_dropout_rate,
                max_len=max_len), )

        self.normalize_before = normalize_before
        self.after_norm = LayerNorm(output_size, epsilon=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate)
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate,
                                   activation)
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)

        self.encoders = nn.LayerList([
            ConformerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(*convolution_layer_args)
                if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after) for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: paddle.Tensor,
            xs_lens: paddle.Tensor,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Embed positions in tensor.
        Args:
            xs: padded input tensor (B, L, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
                the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor, lens and mask
        """
        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks, offset=0)
        # TODO 需要检查这个
        mask_pad = ~masks
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
            att_mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool)
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """ Forward just one chunk
        Args:
            xs (paddle.Tensor): chunk audio feat input, [B=1, T, D], where
                `T==(chunk_size-1)*subsampling_rate + subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache(paddle.Tensor): cache tensor for key & val in
                transformer/conformer attention. Shape is
                (elayers, head, cache_t1, d_k * 2), where`head * d_k == hidden-dim`
                and `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (paddle.Tensor): cache tensor for cnn_module in conformer,
                (elayers, B=1, hidden-dim, cache_t2), where `cache_t2 == cnn.lorder - 1`
        Returns:
            paddle.Tensor: output of current input xs, (B=1, chunk_size, hidden-dim)
            paddle.Tensor: new attention cache required for next chunk, dyanmic shape
                (elayers, head, T, d_k*2) depending on required_cache_size
            paddle.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache
        """
        assert xs.shape[0] == 1  # batch size must be one

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # tmp_masks is just for interface compatibility, [B=1, C=1, T]
        tmp_masks = paddle.ones([1, 1, xs.shape[1]], dtype=paddle.bool)
        # before embed, xs=(B, T, D1), pos_emb=(B=1, T, D)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset=offset)

        _, _, cache_t1, _ = att_cache.shape
        chunk_size = xs.shape[1]
        attention_key_size = cache_t1 + chunk_size

        # only used when using `RelPositionMultiHeadedAttention`
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1, size=attention_key_size)

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache[i:i + 1],
                cnn_cache=cnn_cache[i:i + 1], )
            # new_att_cache = (1, head, attention_key_size, d_k*2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            # new_cnn_cache = (B=1, hidden-dim, cache_t2)
            r_cnn_cache.append(new_cnn_cache)  # add elayer dim

        if self.normalize_before:
            xs = self.after_norm(xs)

        # r_att_cache (elayers, head, T, d_k*2)
        # r_cnn_cache (elayers, B=1, hidden-dim, cache_t2)
        r_att_cache = paddle.concat(r_att_cache, axis=0)
        r_cnn_cache = paddle.stack(r_cnn_cache, axis=0)
        return xs, r_att_cache, r_cnn_cache


class ConformerEncoderLayer(nn.Layer):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            feed_forward: Optional[nn.Layer] = None,
            feed_forward_macaron: Optional[nn.Layer] = None,
            conv_module: Optional[nn.Layer] = None,
            dropout_rate: float = 0.1,
            normalize_before: bool = True,
            concat_after: bool = False, ):
        """Construct an EncoderLayer object.

        Args:
            size (int): Input dimension.
            self_attn (nn.Layer): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward (nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            feed_forward_macaron (nn.Layer): Additional feed-forward module
                instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (nn.Layer): Convolution module instance.
                `ConvlutionModule` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
            concat_after (bool): Whether to concat attention layer's input and
                output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
        """
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size, epsilon=1e-5)  # for the FNN module
        self.norm_mha = LayerNorm(size, epsilon=1e-5)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size, epsilon=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size, epsilon=1e-5)  # for the CNN module
            self.norm_final = LayerNorm(size, epsilon=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
            self,
            x: paddle.Tensor,
            mask: paddle.Tensor,
            pos_emb: paddle.Tensor,
            mask_pad: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
            att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0])
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute encoded features.
        Args:
            x (paddle.Tensor): Input tensor (#batch, time, size).
            mask (paddle.Tensor): Mask tensor for the input (#batch, time, time).
                (0,0,0) means fake mask.
            pos_emb (paddle.Tensor): postional encoding, must not be None
                for ConformerEncoderLayer
            mask_pad (paddle.Tensor): batch padding mask used for conv module.
               (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (paddle.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (paddle.Tensor): Convolution cache in conformer layer
                (1, #batch=1, size, cache_t2). First dim will not be used, just
                for dy2st.
        Returns:
           paddle.Tensor: Output tensor (#batch, time, size).
           paddle.Tensor: Mask tensor (#batch, time, time).
           paddle.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
           paddle.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """
        # (1, #batch=1, size, cache_t2) -> (#batch=1, size, cache_t2)
        cnn_cache = paddle.squeeze(cnn_cache, axis=0)

        # whether to use macaron style FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, cache=att_cache)

        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = paddle.zeros([0, 0, 0], dtype=x.dtype)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
