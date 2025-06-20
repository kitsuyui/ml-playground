import numpy as np
import sympy  # type: ignore


def test_sympy() -> None:
    x = sympy.var("x")
    y = sympy.var("y")

    # quadratic equation
    y = x**2 + 1

    # en: derivative
    # ja: 微分
    dy = sympy.diff(y, x)
    assert dy == 2 * x

    # en: integral
    # ja: 積分
    iy = sympy.integrate(y, x)
    assert iy == x**3 / 3 + x


def test_substitution() -> None:
    # en: substitution
    # ja: 置換 (代入)
    y = sympy.var("y")
    x = sympy.var("x")
    a = sympy.symbols("a")
    b = sympy.symbols("b")
    c = sympy.symbols("c")

    y = a * x**2 + b * x + c
    assert y.subs({a: 1, b: 2, c: 3}) == x**2 + 2 * x + 3
    assert y == a * x**2 + b * x + c

    dy = sympy.diff(y, x)
    assert dy == 2 * a * x + b
    assert dy.subs({a: 1, b: 2, c: 3}) == 2 * x + 2


def test_lambdify() -> None:
    y = sympy.var("y")
    x = sympy.var("x")
    a = sympy.symbols("a")
    b = sympy.symbols("b")
    c = sympy.symbols("c")

    y = a * x**2 + b * x + c
    # en: convert to numpy function
    # ja: numpy 関数に変換
    numpy_func = sympy.lambdify(
        (a, b, c, x),
        y,
        "numpy",
    )
    assert numpy_func(1, 2, 3, 4) == 1 * 4**2 + 2 * 4 + 3
    assert (
        numpy_func(1, 2, 3, np.array([100, 200, 300]))
        == np.array([10203, 40403, 90603])
    ).all()


def test_latex() -> None:
    y = sympy.var("y")
    x = sympy.var("x")
    a = sympy.symbols("a")
    b = sympy.symbols("b")
    c = sympy.symbols("c")

    quadratic = sympy.Eq(y, a * x**2 + b * x + c)
    assert sympy.latex(quadratic) == "y = a x^{2} + b x + c"
