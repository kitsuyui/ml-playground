import pytest

from kitsuyui_ml.legacy.early_stopping import EarlyStopping


def test_early_stopping() -> None:
    es = EarlyStopping(patience=3)
    assert not es.is_stopped()
    assert not es(loss=1.0)  # improved (count reset), not stopped yet
    assert not es.is_best_loss(1.0)  # equal loss is not strictly better
    assert not es.is_stopped()
    assert not es.is_best_loss(1.1)
    assert not es(loss=1.1)  # count=1, not stopped yet
    assert not es.is_stopped()
    assert not es(loss=1.1)  # count=2, not stopped yet
    assert not es.is_stopped()
    assert es(loss=1.1)  # count=3 >= patience=3, stopped
    assert es.is_stopped()


def test_early_stopping_plateau() -> None:
    """Repeated equal losses must count toward patience, not reset it."""
    es = EarlyStopping(patience=3)
    es(loss=1.0)  # best_loss = 1.0
    assert not es.is_best_loss(1.0)  # equal is not strictly better
    assert not es.is_stopped()
    es.step(1.0)  # count = 1
    assert not es.is_stopped()
    es.step(1.0)  # count = 2
    assert not es.is_stopped()
    es.step(1.0)  # count = 3, patience exhausted
    assert es.is_stopped()


def test_early_stopping_nan_loss_raises() -> None:
    es = EarlyStopping(patience=3)
    es(loss=1.0)
    with pytest.raises(ValueError, match="NaN"):
        es(loss=float("nan"))


@pytest.mark.parametrize("loss", [float("inf"), -float("inf")])
def test_early_stopping_infinite_loss_raises_without_resetting_state(
    loss: float,
) -> None:
    es = EarlyStopping(patience=3)
    es(loss=1.0)
    es(loss=1.1)
    count = es.count
    best_loss = es.best_loss

    with pytest.raises(ValueError, match="Infinite"):
        es(loss=loss)

    assert es.count == count
    assert es.best_loss == best_loss


@pytest.mark.parametrize("loss", [float("inf"), -float("inf")])
def test_is_best_loss_rejects_infinite_loss(loss: float) -> None:
    es = EarlyStopping(patience=3)

    with pytest.raises(ValueError, match="Infinite"):
        es.is_best_loss(loss)
