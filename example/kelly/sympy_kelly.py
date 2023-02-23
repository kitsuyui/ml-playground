"""Kelly criterion in Sympy

ja: Sympy での Kelly 基準を扱う
en: Sympy Kelly criterion
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import sympy

ZERO_TO_ONE = sympy.Interval(0.0, 1.0, left_open=False, right_open=False)


@dataclass(frozen=True)
class SympyKelly:
    """SympyKelly

    ja: Sympy での Kelly 基準を扱うクラス 最大損失を割るバージョンで求める (Optimal F と呼ばれるもの)
    en: Sympy Kelly criterion class
    """

    size: int

    @cached_property
    def gains(self) -> Tuple[sympy.Expr, ...]:
        """gains

        ja: 各事象の利得
        en: Gain of each event"""
        return tuple(
            sympy.symbols(f"g_{i}", real=True) for i in range(self.size)
        )

    @cached_property
    def probabilities(self) -> Tuple[sympy.Expr, ...]:
        """probabilities

        ja: 各事象の確率
        en: Probability of each event"""
        return tuple(
            sympy.symbols(
                f"p_{i}",
                real=True,
                positive=True,
                domain=ZERO_TO_ONE,
            )
            for i in range(self.size)
        )

    @cached_property
    def fraction(self) -> sympy.Expr:
        """fraction

        ja: 配分率
        en: Fraction of assets
        """
        return sympy.symbols("f", real=True, positive=True)  # type: ignore

    @cached_property
    def worst_loss(self) -> sympy.Expr:
        """worst_loss

        ja: 最大損失
        en: Worst loss"""
        return -sympy.Min(*self.gains)  # type: ignore

    @cached_property
    def growth(self) -> sympy.Expr:
        """growth

        ja: 資産を f の割合で賭けたときの期待値
        en: Expected value when betting f percent of assets
        """
        g = sympy.Mul(
            *[
                (1 + self.gains[i] * self.fraction / self.worst_loss)
                ** self.probabilities[i]
                for i in range(self.size)
            ]
        )
        return g  # type: ignore
