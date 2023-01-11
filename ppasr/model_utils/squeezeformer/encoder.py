import paddle
import paddle.nn as nn
from typing import Tuple, Union, Optional, List

from typeguard import check_argument_types

from ppasr.model_utils.conformer.attention import MultiHeadedAttention
from ppasr.model_utils.conformer.embedding import RelPositionalEncoding
from ppasr.model_utils.squeezeformer.attention import RelPositionMultiHeadedAttention
from ppasr.model_utils.squeezeformer.convolution import ConvolutionModule
from ppasr.model_utils.squeezeformer.positionwise import PositionwiseFeedForward
from ppasr.model_utils.squeezeformer.subsampling import DepthwiseConv2DSubsampling4
from ppasr.model_utils.squeezeformer.time_reduction import TimeReductionLayer1D, TimeReductionLayerStream, \
    TimeReductionLayer2D
from ppasr.model_utils.utils.base import LayerNorm, Linear
from ppasr.model_utils.utils.common import get_activation
from ppasr.model_utils.utils.mask import make_non_pad_mask, add_optional_chunk_mask

__all__ = ["SqueezeformerEncoder"]


class SqueezeformerEncoder(nn.Layer):
    def __init__(
            self,
            input_size: int,
            encoder_dim: int = 256,
            output_size: int = 256,
            attention_heads: int = 4,
            num_blocks: int = 12,
            reduce_idx: Optional[Union[int, List[int]]] = 5,
            recover_idx: Optional[Union[int, List[int]]] = 11,
            feed_forward_expansion_factor: int = 8,
            dw_stride: bool = False,
            input_dropout_rate: float = 0.1,
            pos_enc_layer_type: str = "rel_pos",
            time_reduction_layer_type: str = "conv1d",
            feed_forward_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            cnn_module_kernel: int = 31,
            cnn_norm_type: str = "layer_norm",
            dropout: float = 0.1,
            causal: bool = False,
            adaptive_scale: bool = True,
            activation_type: str = "swish",
            init_weights: bool = True,
            global_cmvn: paddle.nn.Layer = None,
            normalize_before: bool = False,
            use_dynamic_chunk: bool = False,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_left_chunk: bool = False
    ):
        """Construct SqueezeformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in Transformer BaseEncoder.
            encoder_dim (int): The hidden dimension of encoder layer.
            output_size (int): The output dimension of final projection layer.
            attention_heads (int): Num of attention head in attention module.
            num_blocks (int): Num of encoder layers.
            reduce_idx Optional[Union[int, List[int]]]:
                reduce layer index, from 40ms to 80ms per frame.
            recover_idx Optional[Union[int, List[int]]]:
                recover layer index, from 80ms to 40ms per frame.
            feed_forward_expansion_factor (int): Enlarge coefficient of FFN.
            dw_stride (bool): Whether do depthwise convolution on subsampling module.
            input_dropout_rate (float): Dropout rate of input projection layer.
            pos_enc_layer_type (str): Self attention type.
            time_reduction_layer_type (str): Conv1d or Conv2d reduction layer.
            cnn_module_kernel (int): Kernel size of CNN module.
            activation_type (str): Encoder activation function type.
            cnn_module_kernel (int): Kernel size of convolution module.
            adaptive_scale (bool): Whether to use adaptive scale.
            init_weights (bool): Whether to initialize weights.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__()
        self.global_cmvn = global_cmvn
        self.reduce_idx: Optional[Union[int, List[int]]] = [reduce_idx] \
            if type(reduce_idx) == int else reduce_idx
        self.recover_idx: Optional[Union[int, List[int]]] = [recover_idx] \
            if type(recover_idx) == int else recover_idx
        self.check_ascending_list()
        if reduce_idx is None:
            self.time_reduce = None
        else:
            if recover_idx is None:
                self.time_reduce = 'normal'  # no recovery at the end
            else:
                self.time_reduce = 'recover'  # recovery at the end
                assert len(self.reduce_idx) == len(self.recover_idx)
            self.reduce_stride = 2
        self._output_size = output_size
        self.normalize_before = normalize_before
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads,
                                           output_size,
                                           attention_dropout_rate)
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (attention_heads,
                                           encoder_dim,
                                           attention_dropout_rate,
                                           adaptive_scale,
                                           init_weights)

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (encoder_dim,
                                   encoder_dim * feed_forward_expansion_factor,
                                   feed_forward_dropout_rate,
                                   activation,
                                   adaptive_scale,
                                   init_weights)

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (encoder_dim, cnn_module_kernel, activation,
                                  cnn_norm_type, causal, True, adaptive_scale, init_weights)

        self.embed = DepthwiseConv2DSubsampling4(1, encoder_dim,
                                                 RelPositionalEncoding(encoder_dim, dropout_rate=0.1),
                                                 dw_stride,
                                                 input_size,
                                                 input_dropout_rate,
                                                 init_weights)

        self.preln = LayerNorm(encoder_dim)
        self.encoders = paddle.nn.LayerList([SqueezeformerEncoderLayer(
            encoder_dim,
            encoder_selfattn_layer(*encoder_selfattn_layer_args),
            positionwise_layer(*positionwise_layer_args),
            convolution_layer(*convolution_layer_args),
            positionwise_layer(*positionwise_layer_args),
            normalize_before,
            dropout,
            concat_after) for _ in range(num_blocks)
        ])
        if time_reduction_layer_type == 'conv1d':
            time_reduction_layer = TimeReductionLayer1D
            time_reduction_layer_args = {
                'channel': encoder_dim,
                'out_dim': encoder_dim,
            }
        elif time_reduction_layer_type == 'stream':
            time_reduction_layer = TimeReductionLayerStream
            time_reduction_layer_args = {
                'channel': encoder_dim,
                'out_dim': encoder_dim,
            }
        else:
            time_reduction_layer = TimeReductionLayer2D
            time_reduction_layer_args = {'encoder_dim': encoder_dim}

        self.time_reduction_layer = time_reduction_layer(**time_reduction_layer_args)
        self.time_recover_layer = Linear(encoder_dim, encoder_dim)
        self.final_proj = None
        if output_size != encoder_dim:
            self.final_proj = Linear(encoder_dim, output_size)

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
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        xs_lens = chunk_masks.squeeze(1).sum(1)
        xs = self.preln(xs)
        recover_activations: \
            List[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]] = []
        index = 0
        for i, layer in enumerate(self.encoders):
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append((xs, chunk_masks, pos_emb, mask_pad))
                    xs, xs_lens, chunk_masks, mask_pad = self.time_reduction_layer(xs, xs_lens, chunk_masks, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    index += 1

            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_chunk_masks, recover_pos_emb, recover_mask_pad = recover_activations[index]
                    # recover output length for ctc decode
                    xs = paddle.repeat_interleave(xs, repeats=2, axis=1)
                    xs = self.time_recover_layer(xs)
                    recoverd_t = recover_tensor.shape[1]
                    xs = recover_tensor + xs[:, :recoverd_t, :]
                    chunk_masks = recover_chunk_masks
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad

            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, masks

    def check_ascending_list(self):
        if self.reduce_idx is not None:
            assert self.reduce_idx == sorted(self.reduce_idx), \
                "reduce_idx should be int or ascending list"
        if self.recover_idx is not None:
            assert self.recover_idx == sorted(self.recover_idx), \
                "recover_idx should be int or ascending list"

    def calculate_downsampling_factor(self, i: int) -> int:
        if self.reduce_idx is None:
            return 1
        else:
            reduce_exp, recover_exp = 0, 0
            for exp, rd_idx in enumerate(self.reduce_idx):
                if i >= rd_idx:
                    reduce_exp = exp + 1
            if self.recover_idx is not None:
                for exp, rc_idx in enumerate(self.recover_idx):
                    if i >= rc_idx:
                        recover_exp = exp + 1
            return int(2 ** (reduce_exp - recover_exp))

    def forward_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
            cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
            att_mask: paddle.Tensor = paddle.ones([0, 0, 0], dtype=paddle.bool),
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """ Forward just one chunk

        Args:
            xs (paddle.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (paddle.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (paddle.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            paddle.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            paddle.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            paddle.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.
        """
        assert xs.shape[0] == 1  # batch size must be one

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # tmp_masks is just for interface compatibility, [B=1, C=1, T]
        tmp_masks = paddle.ones([1, 1, xs.shape[1]], dtype=paddle.bool)
        # before embed, xs=(B, T, D1), pos_emb=(B=1, T, D)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset=offset)

        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.shape[0], att_cache.shape[2]
        chunk_size = xs.shape[1]
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1, size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []

        mask_pad = paddle.ones([1, xs.shape[1]], dtype=paddle.bool)
        mask_pad = mask_pad.unsqueeze(1)
        max_att_len: int = 0
        recover_activations: \
            List[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]] = []
        index = 0
        xs_lens = paddle.to_tensor([xs.shape[1]], dtype=paddle.int32)
        xs = self.preln(xs)
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            if self.reduce_idx is not None:
                if self.time_reduce is not None and i in self.reduce_idx:
                    recover_activations.append((xs, att_mask, pos_emb, mask_pad))
                    xs, xs_lens, att_mask, mask_pad = self.time_reduction_layer(xs, xs_lens, att_mask, mask_pad)
                    pos_emb = pos_emb[:, ::2, :]
                    index += 1

            if self.recover_idx is not None:
                if self.time_reduce == 'recover' and i in self.recover_idx:
                    index -= 1
                    recover_tensor, recover_att_mask, recover_pos_emb, recover_mask_pad = recover_activations[index]
                    # recover output length for ctc decode
                    xs = paddle.repeat_interleave(xs, repeats=2, axis=1)
                    xs = self.time_recover_layer(xs)
                    recoverd_t = recover_tensor.shape[1]
                    xs = recover_tensor + xs[:, :recoverd_t, :]
                    att_mask = recover_att_mask
                    pos_emb = recover_pos_emb
                    mask_pad = recover_mask_pad

            factor = self.calculate_downsampling_factor(i)
            att_cache1 = att_cache[i:i + 1][:, :, ::factor, :][:, :, :pos_emb.shape[1] - xs.shape[1], :]
            cnn_cache1 = cnn_cache[i] if cnn_cache.shape[0] > 0 else cnn_cache
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                att_cache=att_cache1,
                cnn_cache=cnn_cache1)
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            cached_att = new_att_cache[:, :, next_cache_start // factor:, :]
            cached_cnn = new_cnn_cache.unsqueeze(0)
            cached_att = cached_att.repeat_interleave(repeats=factor, axis=2)
            if i == 0:
                # record length for the first block as max length
                max_att_len = cached_att.shape[2]
            r_att_cache.append(cached_att[:, :, :max_att_len, :])
            r_cnn_cache.append(cached_cnn)
        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = paddle.concat(r_att_cache, axis=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = paddle.concat(r_cnn_cache, axis=0)

        if self.final_proj is not None:
            xs = self.final_proj(xs)
        return xs, r_att_cache, r_cnn_cache


class SqueezeformerEncoderLayer(nn.Layer):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: paddle.nn.Layer,
            feed_forward1: Optional[nn.Layer] = None,
            conv_module: Optional[nn.Layer] = None,
            feed_forward2: Optional[nn.Layer] = None,
            normalize_before: bool = False,
            dropout_rate: float = 0.1,
            concat_after: bool = False):
        """Construct an EncoderLayer object.

        Args:
            size (int): Input dimension.
            self_attn (paddle.nn.Layer): Self-attention module instance.
                `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
                instance can be used as the argument.
            feed_forward1 (paddle.nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            conv_module (paddle.nn.Layer): Convolution module instance.
                `ConvlutionLayer` instance can be used as the argument.
            feed_forward2 (paddle.nn.Layer): Feed-forward module instance.
                `PositionwiseFeedForward` instance can be used as the argument.
            dropout_rate (float): Dropout rate.
            normalize_before (bool):
                True: use layer_norm before each sub-block.
                False: use layer_norm after each sub-block.
        """
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.layer_norm1 = LayerNorm(size)
        self.ffn1 = feed_forward1
        self.layer_norm2 = LayerNorm(size)
        self.conv_module = conv_module
        self.layer_norm3 = LayerNorm(size)
        self.ffn2 = feed_forward2
        self.layer_norm4 = LayerNorm(size)
        self.normalize_before = normalize_before
        self.dropout = nn.Dropout(dropout_rate)
        self.concat_after = concat_after
        if concat_after:
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
            cnn_cache: paddle.Tensor = paddle.zeros([0, 0, 0, 0]),
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
        # self attention module
        residual = x
        if self.normalize_before:
            x = self.layer_norm1(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.layer_norm1(x)

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm2(x)
        x = self.ffn1(x)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm2(x)

        # conv module
        residual = x
        if self.normalize_before:
            x = self.layer_norm3(x)
        x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm3(x)

        # ffn module
        residual = x
        if self.normalize_before:
            x = self.layer_norm4(x)
        x = self.ffn2(x)
        # we do not use dropout here since it is inside feed forward function
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.layer_norm4(x)

        return x, mask, new_att_cache, new_cnn_cache
