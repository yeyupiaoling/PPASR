import paddle


class GlobalCMVN(paddle.nn.Layer):
    def __init__(self,
                 mean: paddle.Tensor,
                 istd: paddle.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (paddle.Tensor): mean stats
            istd (paddle.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: paddle.Tensor):
        """
        Args:
            x (paddle.Tensor): (batch, max_len, feat_dim)

        Returns:
            (paddle.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
