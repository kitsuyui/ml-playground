import pytest
import torch

from kitsuyui_ml.legacy.torch.scheduler import LRScheduler


def test_scheduler() -> None:
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.1)
    scheduler: LRScheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 0.1)
    assert pytest.approx(scheduler.get_last_lr()) == [0.01]
