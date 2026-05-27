import datetime

import pytest

from kitsuyui_ml.legacy.epoch_result import EpochResult

_UTC = datetime.timezone.utc


def make_epoch_result(
    *,
    valid_loss: float = 0.2,
    best_loss: float = 0.2,
) -> EpochResult:
    return EpochResult(
        epoch=1,
        epochs=10,
        train_size=100,
        valid_size=50,
        train_loss=0.1,
        valid_loss=valid_loss,
        best_loss=best_loss,
        early_stop_count=0,
        early_stop_patience=5,
        begin_train=datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=_UTC),
        end_train=datetime.datetime(2020, 1, 1, 0, 0, 1, tzinfo=_UTC),
        begin_valid=datetime.datetime(2020, 1, 1, 0, 0, 1, tzinfo=_UTC),
        end_valid=datetime.datetime(2020, 1, 1, 0, 0, 2, tzinfo=_UTC),
    )


def test_epoch_result() -> None:
    er = make_epoch_result()
    assert er.epoch == 1
    assert er.epochs == 10
    assert er.train_size == 100
    assert er.valid_size == 50
    assert er.train_loss == 0.1
    assert er.valid_loss == 0.2
    assert er.best_loss == 0.2
    assert er.early_stop_count == 0
    assert er.early_stop_patience == 5
    assert er.begin_train == datetime.datetime(
        2020, 1, 1, 0, 0, 0, tzinfo=_UTC
    )

    assert er.train_time == datetime.timedelta(seconds=1)
    assert er.valid_time == datetime.timedelta(seconds=1)
    assert er.total_time == datetime.timedelta(seconds=2)


def test_epoch_result_is_best_loss_requires_improvement() -> None:
    assert make_epoch_result(valid_loss=0.1, best_loss=0.2).is_best_loss()
    assert not make_epoch_result(valid_loss=0.2, best_loss=0.2).is_best_loss()
    assert not make_epoch_result(valid_loss=0.3, best_loss=0.2).is_best_loss()


def _make_with_override(field: str, value: datetime.datetime) -> EpochResult:
    defaults = {
        "begin_train": datetime.datetime(2020, 1, 1, 0, 0, 0, tzinfo=_UTC),
        "end_train": datetime.datetime(2020, 1, 1, 0, 0, 1, tzinfo=_UTC),
        "begin_valid": datetime.datetime(2020, 1, 1, 0, 0, 1, tzinfo=_UTC),
        "end_valid": datetime.datetime(2020, 1, 1, 0, 0, 2, tzinfo=_UTC),
    }
    defaults[field] = value
    return EpochResult(
        epoch=1,
        epochs=10,
        train_size=100,
        valid_size=50,
        train_loss=0.1,
        valid_loss=0.2,
        best_loss=0.2,
        early_stop_count=0,
        early_stop_patience=5,
        begin_train=defaults["begin_train"],
        end_train=defaults["end_train"],
        begin_valid=defaults["begin_valid"],
        end_valid=defaults["end_valid"],
    )


def test_epoch_result_rejects_naive_datetime() -> None:
    naive = datetime.datetime(2020, 1, 1, 0, 0, 0)
    for field in ("begin_train", "end_train", "begin_valid", "end_valid"):
        with pytest.raises(ValueError, match="timezone-aware"):
            _make_with_override(field, naive)
