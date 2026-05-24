import pytest

from kitsuyui_ml.legacy.early_stopping import EarlyStopping


def test_early_stopping() -> None:
    es = EarlyStopping(patience=3)
    assert not es.is_stopped()
    assert not es(loss=1.0)  # improved, not stopped yet
    assert es.is_best_loss(1.0)
    assert not es.is_stopped()
    assert not es.is_best_loss(1.1)
    assert not es(loss=1.1)  # count=1, not stopped yet
    assert not es.is_stopped()
    assert not es(loss=1.1)  # count=2, not stopped yet
    assert not es.is_stopped()
    assert es(loss=1.1)  # count=3 >= patience=3, stopped
    assert es.is_stopped()


def test_early_stopping_nan_loss_raises() -> None:
    es = EarlyStopping(patience=3)
    es(loss=1.0)
    with pytest.raises(ValueError, match="NaN"):
        es(loss=float("nan"))
