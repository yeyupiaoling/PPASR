from typing import Union

from paddle.optimizer.lr import LRScheduler
from typeguard import check_argument_types


class WarmupLR(LRScheduler):
    """The WarmupLR scheduler
    This scheduler is almost same as NoamLR Scheduler except for following
    difference:
    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(self,
                 warmup_steps: Union[int, float] = 25000,
                 learning_rate=1.0,
                 last_epoch=-1,
                 verbose=False,
                 **kwargs):
        assert check_argument_types()
        self.warmup_steps = warmup_steps
        super().__init__(learning_rate, last_epoch, verbose)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, lr={self.base_lr}, last_epoch={self.last_epoch})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        if self.warmup_steps == 0:
            return self.base_lr * step_num ** -0.5
        else:
            return self.base_lr * self.warmup_steps ** 0.5 * min(
                step_num ** -0.5, step_num * self.warmup_steps ** -1.5)

    def set_step(self, step: int = None):
        '''
        It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .

        Args:
            step (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        '''
        self.step(epoch=step)
