from typing import Union

import torch

# treat all scheduler classes as a single type.
LRScheduler = Union[
    torch.optim.lr_scheduler.ConstantLR,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.CyclicLR,
    torch.optim.lr_scheduler.ExponentialLR,
    torch.optim.lr_scheduler.LambdaLR,
    torch.optim.lr_scheduler.LinearLR,
    torch.optim.lr_scheduler.MultiStepLR,
    torch.optim.lr_scheduler.MultiplicativeLR,
    torch.optim.lr_scheduler.OneCycleLR,
    torch.optim.lr_scheduler.PolynomialLR,
    torch.optim.lr_scheduler.SequentialLR,
    torch.optim.lr_scheduler.StepLR,
]
