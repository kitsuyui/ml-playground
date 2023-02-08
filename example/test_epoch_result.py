import datetime

from example.epoch_result import EpochResult


def test_epoch_result() -> None:
    er = EpochResult(
        epoch=1,
        epochs=10,
        train_size=100,
        valid_size=50,
        train_loss=0.1,
        valid_loss=0.2,
        best_loss=0.2,
        early_stop_count=0,
        early_stop_patience=5,
        begin_train=datetime.datetime(2020, 1, 1, 0, 0, 0),
        end_train=datetime.datetime(2020, 1, 1, 0, 0, 1),
        begin_valid=datetime.datetime(2020, 1, 1, 0, 0, 1),
        end_valid=datetime.datetime(2020, 1, 1, 0, 0, 2),
    )
    assert er.epoch == 1
    assert er.epochs == 10
    assert er.train_size == 100
    assert er.valid_size == 50
    assert er.train_loss == 0.1
    assert er.valid_loss == 0.2
    assert er.best_loss == 0.2
    assert er.early_stop_count == 0
    assert er.early_stop_patience == 5
    assert er.begin_train == datetime.datetime(2020, 1, 1, 0, 0, 0)

    assert er.train_time == datetime.timedelta(seconds=1)
    assert er.valid_time == datetime.timedelta(seconds=1)
    assert er.total_time == datetime.timedelta(seconds=2)

    assert er.is_best_loss()
