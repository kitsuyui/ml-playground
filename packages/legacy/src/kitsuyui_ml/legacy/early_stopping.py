import math
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Early stopping class to stop training when loss is not improving."""

    patience: int
    count: int = 0
    best_loss: float = float("inf")

    def __call__(self, loss: float) -> bool:
        self._validate_loss(loss)
        is_best_loss = self._is_best_loss_unchecked(loss)
        self._step_unchecked(loss)
        return is_best_loss

    @staticmethod
    def _validate_loss(loss: float) -> None:
        if math.isnan(loss):
            raise ValueError("NaN loss: possible gradient explosion")
        if math.isinf(loss):
            raise ValueError("Infinite loss: possible gradient explosion")

    def is_best_loss(self, loss: float) -> bool:
        """Check if loss is better than current best loss."""
        self._validate_loss(loss)
        return self._is_best_loss_unchecked(loss)

    def _is_best_loss_unchecked(self, loss: float) -> bool:
        return loss <= self.best_loss

    def step(self, loss: float) -> None:
        """Update early stopping state."""
        self._validate_loss(loss)
        self._step_unchecked(loss)

    def _step_unchecked(self, loss: float) -> None:
        if self._is_best_loss_unchecked(loss):
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1

    def is_stopped(self) -> bool:
        """Check if early stopping is triggered."""
        return self.count >= self.patience
