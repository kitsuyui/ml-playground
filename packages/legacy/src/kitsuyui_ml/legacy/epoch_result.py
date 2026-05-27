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

    def __post_init__(self) -> None:
        for field_name in (
            "begin_train",
            "end_train",
            "begin_valid",
            "end_valid",
        ):
            value: datetime.datetime = getattr(self, field_name)
            if value.tzinfo is None:
                raise ValueError(
                    f"{field_name} must be timezone-aware; got naive datetime. "
                    "Use datetime.now(tz=datetime.timezone.utc) or attach tzinfo."
                )

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
        return self.valid_loss < self.best_loss
