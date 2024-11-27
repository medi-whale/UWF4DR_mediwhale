import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class CosineWarmup(LambdaLR):

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, end_step=-1, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.end_step = end_step

        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1.0, self.warmup_steps))
            else:
                return 0.5 * (
                    1.0
                    + math.cos(
                        math.pi * (step - self.warmup_steps) / (self.end_step - self.warmup_steps)
                    )
                )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)