"""Kelly

ja: ケリー基準
en: Kelly criterion"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Tuple

import scipy
import scipy.optimize
import sympy

from .sympy_kelly import SympyKelly

Gain = float
Probability = float
GainProbabilityTable = List[Tuple[Gain, Probability]]


@dataclass
class KellySolver:
    """KellySolver

    ja: ケリー基準を解く
    en: Solve Kelly criterion"""

    table: GainProbabilityTable
    assets: float
    _sympy: SympyKelly = field(init=False)

    def __post_init__(self) -> None:
        if sum(self.probabilities) != 1.0:
            raise ValueError("sum of probabilities must be 1.0")
        if min(self.gains) > 0:
            raise ValueError("all positive gains are meaningless")
        self._sympy = SympyKelly(self.size)

    @cached_property
    def gains(self) -> Tuple[float, ...]:
        """gains

        ja: 各事象の利得
        en: Gain of each event"""
        return tuple(gain for gain, _ in self.table)

    @cached_property
    def probabilities(self) -> Tuple[float, ...]:
        """probabilities

        ja: 各事象の確率
        en: Probability of each event"""
        return tuple(prob for _, prob in self.table)

    @cached_property
    def size(self) -> int:
        """size

        ja: 事象の数
        en: Number of events"""
        return len(self.table)

    @cached_property
    def worst_loss(self) -> float:
        """worst_loss

        ja: 最大損失
        en: Worst loss"""
        return float(self._sympy.worst_loss.subs(self.variables_to_real))

    @cached_property
    def variables_to_real(self) -> Dict[sympy.Expr, float]:
        """variables_to_real

        ja: Scipy の変数を実際の値に置き換えるときの変数
        en: Variables to replace scipy variables with real values
        """
        return {
            **dict(zip(self._sympy.probabilities, self.probabilities)),
            **dict(zip(self._sympy.gains, self.gains)),
        }

    @cached_property
    def optimal_fraction(self) -> float:
        """optimal_fraction

        ja: 資産の配分の割合 (ただし最大損失の調整が入っている)
        en: Fraction of asset allocation (but adjusted for maximum loss)
        """
        return arg_max_solver(
            variable=self._sympy.fraction,
            function=self._sympy.growth.subs(self.variables_to_real),
        )

    @cached_property
    def optimal_bets(self) -> float:
        """optimal_bets

        ja: 最適な資産の配分 (ただし最大損失の調整が入っている)
        en: Optimal asset allocation (but adjusted for maximum loss)
        """
        return self.optimal_fraction * self.assets

    @cached_property
    def optimal_bet_units(self) -> float:
        """optimal_bet_units

        ja: 最適なべットの単位
        en: Optimal bet unit
        """
        return self.optimal_bets / self.worst_loss

    @cached_property
    def optimal_growth(self) -> float:
        """optimal_growth

        ja: 最適なべットをした場合の成長率
        en: Growth rate when optimal bet is made
        """
        variables = {
            **self.variables_to_real,
            self._sympy.fraction: self.optimal_fraction,
        }
        return float(self._sympy.growth.subs(variables))


def arg_max_solver(*, variable: sympy.Expr, function: sympy.Expr) -> float:
    """arg_max_solver

    ja: Sympy を scipy に変換して最大値を持つ変数を求める
    en: Convert sympy to scipy and find the variable with the maximum value
    """
    function_in_scipy = sympy.lambdify(
        (variable,),
        -function,  # inversed maximize is minimize
        "scipy",
    )
    arg_max_values = scipy.optimize.minimize(function_in_scipy, 0).x
    return float(arg_max_values[0])  # scipy -> float
