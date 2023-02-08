from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Early stopping class to stop training when loss is not improving."""

    patience: int
    count: int = 0
    best_loss: float = float("inf")

    def __call__(self, loss: float) -> bool:
        is_best_loss = self.is_best_loss(loss)
        self.step(loss)
        return is_best_loss

    def is_best_loss(self, loss: float) -> bool:
        """Check if loss is better than current best loss."""
        return loss <= self.best_loss

    def step(self, loss: float) -> None:
        """Update early stopping state."""
        if self.is_best_loss(loss):
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1

    def is_stopped(self) -> bool:
        """Check if early stopping is triggered."""
        return self.count >= self.patience
