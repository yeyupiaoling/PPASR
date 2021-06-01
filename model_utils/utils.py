import paddle

__all__ = ['brelu', 'make_non_pad_mask']


def brelu(x, t_min=0.0, t_max=24.0):
    t_min = paddle.full(shape=[1], fill_value=t_min, dtype=paddle.float32)
    t_max = paddle.full(shape=[1], fill_value=t_max, dtype=paddle.float32)
    return x.maximum(t_min).minimum(t_max)


def make_non_pad_mask(lengths: paddle.Tensor) -> paddle.Tensor:
    batch_size = int(lengths.shape[0])
    max_len = int(lengths.max())
    seq_range = paddle.arange(0, max_len, dtype=paddle.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len])
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask.logical_not()
