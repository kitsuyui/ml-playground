import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    epochs: int
    train_size: int
    valid_size: int
    train_loss: float
    valid_loss: float
    best_loss: float
    early_stop_count: int
    early_stop_patience: int
    begin_train: datetime.datetime
    end_train: datetime.datetime
    begin_valid: datetime.datetime
    end_valid: datetime.datetime

    @property
    def train_time(self) -> datetime.timedelta:
        return self.end_train - self.begin_train

    @property
    def valid_time(self) -> datetime.timedelta:
        return self.end_valid - self.begin_valid

    @property
    def total_time(self) -> datetime.timedelta:
        return self.end_valid - self.begin_train

    def is_best_loss(self) -> bool:
        return self.valid_loss <= self.best_loss
