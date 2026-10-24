"""Finite-difference gradient checks for every differentiable operation."""
import numpy as np
import pytest

from engine import Tensor, softmax_cross_entropy
from engine.gradcheck import check_gradients

LABELS = np.array([0, 2, 1, 2, 0, 1])


def _inputs(shapes, seed):
    rng = np.random.default_rng(seed)
    tensors = []
    for shape in shapes:
        x = rng.normal(size=shape)
        # Keep values away from 0: ReLU's kink and division by ~0 both make
        # central differences meaningless there.
        x = x + 0.2 * np.sign(x)
        tensors.append(Tensor(x, requires_grad=True))
    return tensors


def _scalarize(out, seed=123):
    """Reduce a non-scalar output with distinct random weights so every output entry is checked."""
    if out.data.size == 1:
        return out.sum()
    weights = np.random.default_rng(seed).normal(size=out.shape)
    return (out * weights).sum()


CASES = [
    # element-wise
    ("add", [(3, 4), (3, 4)], lambda a, b: a + b),
    ("sub", [(3, 4), (3, 4)], lambda a, b: a - b),
    ("mul", [(3, 4), (3, 4)], lambda a, b: a * b),
    ("div", [(3, 4), (3, 4)], lambda a, b: a / b),
    ("neg", [(3, 4)], lambda a: -a),
    ("scalar radd", [(3, 4)], lambda a: 2.0 + a),
    ("scalar rsub", [(3, 4)], lambda a: 1.0 - a),
    ("scalar rmul", [(3, 4)], lambda a: 3.0 * a),
    ("scalar rdiv", [(3, 4)], lambda a: 1.0 / a),
    # broadcasting
    ("(N,D) + (D,)", [(4, 3), (3,)], lambda x, b: x + b),
    ("(D,) - (N,D)", [(3,), (4, 3)], lambda b, x: b - x),
    ("(N,1) * (1,D)", [(4, 1), (1, 3)], lambda a, b: a * b),
    ("(N,D) / (D,)", [(4, 3), (3,)], lambda x, b: x / b),
    ("(B,N,D) * (N,1)", [(2, 4, 3), (4, 1)], lambda x, b: x * b),
    ("(N,D) + (1,)", [(4, 3), (1,)], lambda x, b: x + b),
    # linear algebra
    ("matmul", [(5, 3), (3, 4)], lambda a, b: a @ b),
    ("batched matmul", [(2, 5, 3), (2, 3, 4)], lambda a, b: a @ b),
    ("broadcast matmul", [(2, 5, 3), (3, 4)], lambda a, b: a @ b),
    # reductions
    ("sum all", [(3, 4)], lambda a: a.sum()),
    ("sum axis 0", [(3, 4)], lambda a: a.sum(axis=0)),
    ("sum axis 1 keepdims", [(3, 4)], lambda a: a.sum(axis=1, keepdims=True)),
    ("sum negative axes", [(2, 3, 4)], lambda a: a.sum(axis=(-1, 0))),
    ("mean all", [(3, 4)], lambda a: a.mean()),
    ("mean axis 0", [(3, 4)], lambda a: a.mean(axis=0)),
    ("mean axis 1 keepdims", [(2, 3, 4)], lambda a: a.mean(axis=1, keepdims=True)),
    # non-linearities
    ("relu", [(3, 4)], lambda a: a.relu()),
    ("tanh", [(3, 4)], lambda a: a.tanh()),
    ("sigmoid", [(3, 4)], lambda a: a.sigmoid()),
    # loss
    ("softmax cross-entropy", [(6, 3)], lambda z: softmax_cross_entropy(z, LABELS)),
    # graph structure: re-used nodes and a diamond
    ("reuse x*x + x", [(3, 3)], lambda a: a * a + a),
    ("diamond", [(3, 3)], lambda a: (a.tanh() * a.sigmoid()) / (1.0 + a * a)),
    (
        "two-layer MLP + CE",
        [(6, 4), (4, 5), (5,), (5, 3), (3,)],
        lambda x, w1, b1, w2, b2: softmax_cross_entropy(((x @ w1 + b1).relu()) @ w2 + b2, LABELS),
    ),
]


@pytest.mark.parametrize("name, shapes, fn", CASES, ids=[c[0] for c in CASES])
def test_gradient_matches_finite_differences(name, shapes, fn):
    inputs = _inputs(shapes, seed=sum(map(ord, name)))
    check_gradients(lambda: _scalarize(fn(*inputs)), inputs)
