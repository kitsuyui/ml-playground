from example.early_stopping import EarlyStopping


def test_early_stopping() -> None:
    es = EarlyStopping(patience=3)
    assert not es.is_stopped()
    assert es(loss=1.0)
    assert es.is_best_loss(1.0)
    assert not es.is_stopped()
    assert not es.is_best_loss(1.1)
    assert not es(loss=1.1)
    assert not es.is_stopped()
    assert not es(loss=1.1)
    assert not es.is_stopped()
    assert not es(loss=1.1)
    assert es.is_stopped()
