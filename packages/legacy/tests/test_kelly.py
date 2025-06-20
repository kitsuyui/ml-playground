"""ケリー基準の計算のテスト"""

import pytest

from kitsuyui_ml.legacy.kelly.kelly import KellySolver


def test_kelly() -> None:
    """Test for KellySolver"""
    # 花子さんの口座残高は 200万円。ソニー専門で売買している。
    # 1株で売買するとした場合、損益の確率は、以下のようになっているという。
    # ・ 300円の利益 … 確率は 10％
    # ・ 200円の利益 … 確率は 15％
    # ・ 150円の利益 … 確率は 30％
    # ・ 200円の損失 … 確率は 35％
    # ・ 250円の損失 … 確率は 10％
    # さて、ソニーに買いシグナルが出た。株価は 4000 円である。ケリー基準に従うとして、何株買うべきか？
    # from http://geolog.mydns.jp/www.geocities.jp/y_infty/management/criterion_2.html  # noqa
    kelly = KellySolver(
        table=[
            (+300, 0.10),
            (+200, 0.15),
            (+150, 0.30),
            (-200, 0.35),
            (-250, 0.10),
        ],
        assets=2_000_000,
    )

    assert kelly.optimal_fraction == pytest.approx(0.0595, 0.001), "Optimal f ≒ 0.0595"
    assert kelly.optimal_bets == pytest.approx(119_075, 0.1), (
        "119,075 円まで買うのが正解"
    )
    assert kelly.optimal_bet_units == pytest.approx(476, 0.1), "476 株買うのが正解"
    assert kelly.optimal_growth == pytest.approx(1.00119, 1e-5), "0.119% の複利で増える"


def test_kelly_error() -> None:
    """Test for KellySolver"""
    # ja: そもそも常に儲かるなら ValueError
    # en: ValueError if always profitable
    with pytest.raises(ValueError) as error:
        KellySolver(
            [
                (1, 0.6),
                (1, 0.4),
            ],
            10000.0,
        )
    assert str(error.value) == "all positive gains are meaningless"

    # ja: 確率の合計が 1 じゃないなら ValueError
    # en: ValueError if sum of probabilities is not 1
    with pytest.raises(ValueError) as error:
        KellySolver(
            [
                (1, 0.6),
                (-1, 0.3),
            ],
            10000.0,
        )
    assert str(error.value) == "sum of probabilities must be 1.0"
