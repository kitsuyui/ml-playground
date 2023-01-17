import numpy as np
import pytest
import sympy
import torch
import torch.nn as nn


def test_nn_linear() -> None:
    """linear

    y = Wx + b (W: weight, b: bias)

    a.k.a.
    - affine transformation
    - dense layer
    - fully connected layer
    """
    minimal_linear = nn.Linear(in_features=3, out_features=2)
    with torch.no_grad():
        minimal_linear.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        )
        minimal_linear.bias.copy_(torch.tensor([0.0, 0.0]))
        actual = minimal_linear(torch.tensor([1.0, 2.0, 3.0]))

    tobe = torch.tensor([14.0, 20.0])
    assert torch.allclose(actual, tobe)

    # numpy version
    weight = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    bias = np.array([0.0, 0.0])
    input_n = np.array([1.0, 2.0, 3.0])
    actual_n = np.matmul(weight, input_n) + bias
    tobe_n = np.array([14.0, 20.0])
    assert np.allclose(actual_n, tobe_n)


def test_nn_relu() -> None:
    """ReLU

    y = max(0, x)

    a.k.a.
    - rectified linear unit
    """
    relu = nn.ReLU()
    actual = relu(torch.tensor([-1.0, 0.0, 1.0]))
    tobe = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(actual, tobe)

    # numpy version
    input_n = np.array([-1.0, 0.0, 1.0])
    actual_n = np.maximum(0, input_n)
    tobe_n = np.array([0.0, 0.0, 1.0])
    assert np.allclose(actual_n, tobe_n)

    # sympy version
    x = sympy.symbols("x")
    y = sympy.Max(0, x)
    assert sympy.latex(y) == r"\max\left(0, x\right)"
    assert y.subs(x, -1.0) == 0.0
    assert y.subs(x, 0.0) == 0.0
    assert y.subs(x, 1.0) == 1.0
    assert sympy.limit(y, x, sympy.Float("-inf")) == 0.0
    assert sympy.limit(y, x, sympy.Float("inf")) == sympy.Float("inf")


def test_nn_sigmoid() -> None:
    """sigmoid

    y = 1 / (1 + exp(-x))

    a.k.a.
    - logistic function
    """
    sigmoid = nn.Sigmoid()
    actual = sigmoid(torch.tensor([-1.0, 0.0, 1.0]))
    tobe = torch.tensor([0.26894143, 0.5, 0.7310586])
    assert torch.allclose(actual, tobe)

    # numpy version
    input_n = np.array([-1.0, 0.0, 1.0])
    actual_n = 1 / (1 + np.exp(-input_n))
    tobe_n = np.array([0.26894143, 0.5, 0.7310586])
    assert np.allclose(actual_n, tobe_n)

    # sympy version
    x = sympy.symbols("x")
    y = 1 / (1 + sympy.exp(-x))
    assert sympy.latex(y) == r"\frac{1}{1 + e^{- x}}"
    assert pytest.approx(y.subs(x, -1.0)) == 0.268941421369995
    assert pytest.approx(y.subs(x, 0.0)) == 0.5
    assert pytest.approx(y.subs(x, 1.0)) == 0.731058578630005
    assert sympy.limit(y, x, sympy.Float("-inf")) == 0.0
    assert sympy.limit(y, x, sympy.Float("inf")) == 1.0


def test_nn_tanh() -> None:
    """tanh

    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    a.k.a.
    - hyperbolic tangent
    """
    tanh = nn.Tanh()
    actual = tanh(torch.tensor([-1.0, 0.0, 1.0]))
    tobe = torch.tensor([-0.7615942, 0.0, 0.7615942])
    assert torch.allclose(actual, tobe)

    # numpy version
    input_n = np.array([-1.0, 0.0, 1.0])
    actual_n = (np.exp(input_n) - np.exp(-input_n)) / (
        np.exp(input_n) + np.exp(-input_n)
    )
    tobe_n = np.array([-0.7615942, 0.0, 0.7615942])
    assert np.allclose(actual_n, tobe_n)

    # sympy version
    x = sympy.symbols("x")
    y = (sympy.exp(x) - sympy.exp(-x)) / (sympy.exp(x) + sympy.exp(-x))
    assert pytest.approx(y.subs(x, -1.0)) == -0.761594155955765
    assert pytest.approx(y.subs(x, 0.0)) == 0.0
    assert pytest.approx(y.subs(x, 1.0)) == 0.761594155955765
    assert sympy.limit(y, x, sympy.Float("-inf")) == -1.0
    assert sympy.limit(y, x, sympy.Float("inf")) == 1.0
