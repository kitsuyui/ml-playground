import math
from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Early stopping class to stop training when loss is not improving."""

    patience: int
    _count: int = field(default=0, init=False, repr=False)
    _best_loss: float = field(default=float("inf"), init=False, repr=False)

    @property
    def count(self) -> int:
        return self._count

    @property
    def best_loss(self) -> float:
        return self._best_loss

    def __call__(self, loss: float) -> bool:
        """Return True when training should stop (i.e. patience is exhausted)."""
        self.step(loss)
        return self.is_stopped()

    @staticmethod
    def _validate_loss(loss: float) -> None:
        if math.isnan(loss):
            raise ValueError("NaN loss: possible gradient explosion")
        if math.isinf(loss):
            raise ValueError("Infinite loss: possible gradient explosion")

    def is_best_loss(self, loss: float) -> bool:
        """Return True only when loss is strictly better than the current best."""
        self._validate_loss(loss)
        return self._is_best_loss_unchecked(loss)

    def _is_best_loss_unchecked(self, loss: float) -> bool:
        return loss < self._best_loss

    def step(self, loss: float) -> None:
        """Update early stopping state."""
        self._validate_loss(loss)
        self._step_unchecked(loss)

    def _step_unchecked(self, loss: float) -> None:
        if self._is_best_loss_unchecked(loss):
            self._best_loss = loss
            self._count = 0
        else:
            self._count += 1

    def is_stopped(self) -> bool:
        """Check if early stopping is triggered."""
        return self._count >= self.patience
