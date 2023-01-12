import paddle

__all__ = [
    "make_xs_mask", "make_pad_mask", "make_non_pad_mask", "subsequent_mask",
    "subsequent_chunk_mask", "add_optional_chunk_mask"
]


def make_xs_mask(xs: paddle.Tensor, pad_value=0.0) -> paddle.Tensor:
    """Maks mask tensor containing indices of non-padded part.
    Args:
        xs (paddle.Tensor): (B, T, D), zeros for pad.
    Returns:
        paddle.Tensor: Mask Tensor indices of non-padded part. (B, T)
    """
    pad_frame = paddle.full([1, 1, xs.shape[-1]], pad_value, dtype=xs.dtype)
    mask = xs != pad_frame
    mask = mask.all(axis=-1)
    return mask


def make_pad_mask(lengths: paddle.Tensor) -> paddle.Tensor:
    """Make mask tensor containing indices of padded part.
    See description of make_non_pad_mask.
    Args:
        lengths (paddle.Tensor): Batch of lengths (B,).
    Returns:
        paddle.Tensor: Mask tensor containing indices of padded part.
        (B, T)
    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = int(lengths.shape[0])
    max_len = int(lengths.max())
    seq_range = paddle.arange(0, max_len, dtype=paddle.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len])
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: paddle.Tensor) -> paddle.Tensor:
    """Make mask tensor containing indices of non-padded part.
    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.
    This pad_mask is used in both encoder and decoder.
    1 for non-padded part and 0 for padded part.
    Args:
        lengths (paddle.Tensor): Batch of lengths (B,).
    Returns:
        paddle.Tensor: mask tensor containing indices of padded part.
        (B, T)
    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    # return ~make_pad_mask(lengths)
    return make_pad_mask(lengths).logical_not()


def subsequent_mask(size: int) -> paddle.Tensor:
    """Create mask for subsequent steps (size, size).
    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.
    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this case, no attention mask is needed.
    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.
    Args:
        size (int): size of mask
    Returns:
        paddle.Tensor: mask, [size, size]
    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = paddle.ones([size, size], dtype=paddle.bool)
    return paddle.tril(ret)


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1, ) -> paddle.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
    Returns:
        paddle.Tensor: mask, [size, size]
    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = paddle.zeros([size, size], dtype=paddle.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max(0, (i // chunk_size - num_left_chunks) * chunk_size)
        ending = min(size, (i // chunk_size + 1) * chunk_size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: paddle.Tensor,
                            masks: paddle.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int):
    """ Apply optional mask for encoder.
    Args:
        xs (paddle.Tensor): padded input, (B, L, D), L for max length
        mask (paddle.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks (int): number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
    Returns:
        paddle.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = int(paddle.randint(1, max_len, (1,)))
            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = int(paddle.randint(0, max_left_chunks, (1,)))
        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size, num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        # chunk_masks = masks & chunk_masks  # (B, L, L)
        chunk_masks = masks.logical_and(chunk_masks)  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.shape[1], static_chunk_size, num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        # chunk_masks = masks & chunk_masks  # (B, L, L)
        chunk_masks = masks.logical_and(chunk_masks)  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks
