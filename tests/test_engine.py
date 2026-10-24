"""Graph mechanics: accumulation, ordering, broadcasting, no_grad, stability."""
import numpy as np
import pytest

from engine import Tensor, no_grad, softmax_cross_entropy


def test_reused_node_accumulates_every_path():
    x = Tensor([2.0, -3.0], requires_grad=True)
    (x * x * x).sum().backward()
    np.testing.assert_allclose(x.grad, 3 * x.data**2)


def test_diamond_graph_visits_shared_node_once():
    # y = a + a where a = 2x; a must push its gradient only after both uses
    # have reached it, giving dy/dx = 4 (not 2, and not 8).
    x = Tensor(1.5, requires_grad=True)
    a = x * 2.0
    (a + a).backward()
    assert x.grad == pytest.approx(4.0)


def test_leaf_grads_accumulate_across_backward_calls_until_zeroed():
    x = Tensor([1.0, 2.0], requires_grad=True)
    (x * 3.0).sum().backward()
    (x * 3.0).sum().backward()
    np.testing.assert_allclose(x.grad, [6.0, 6.0])
    x.zero_grad()
    assert x.grad is None


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor(1.0, requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_bias_broadcast_gradient_is_column_sum():
    x = Tensor(np.arange(12.0).reshape(4, 3), requires_grad=True)
    b = Tensor(np.zeros(3), requires_grad=True)
    upstream = np.random.default_rng(0).normal(size=(4, 3))
    (x + b).backward(upstream)
    assert b.grad.shape == (3,)
    np.testing.assert_allclose(b.grad, upstream.sum(axis=0))
    np.testing.assert_allclose(x.grad, upstream)


def test_non_scalar_backward_needs_explicit_grad():
    with pytest.raises(RuntimeError):
        (Tensor([1.0, 2.0], requires_grad=True) * 2.0).backward()


def test_no_grad_records_no_graph():
    w = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        y = (w * 3.0).sum()
    assert not y.requires_grad and y._parents == ()


def test_constants_receive_no_gradient():
    x = Tensor([1.0, 2.0])
    w = Tensor([3.0, 4.0], requires_grad=True)
    (x * w).sum().backward()
    assert x.grad is None


def test_float32_is_preserved():
    w = Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
    out = (w @ w).sum()
    out.backward()
    assert out.dtype == np.float32 and w.grad.dtype == np.float32


def test_softmax_cross_entropy_is_stable_for_extreme_logits():
    logits = Tensor([[1e4, -1e4, 0.0], [-5e3, 5e3, 5e3]], requires_grad=True)
    loss = softmax_cross_entropy(logits, [0, 1])
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(logits.grad))
    # Row 0 is a confident correct prediction, row 1 is a 50/50 tie: loss = log(2) / 2.
    assert loss.item() == pytest.approx(np.log(2) / 2)


def test_softmax_cross_entropy_matches_naive_formula_and_is_shift_invariant():
    z = np.random.default_rng(1).normal(size=(5, 4))
    y = np.array([0, 3, 1, 2, 3])
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    naive = -np.log(probs[np.arange(5), y]).mean()
    assert softmax_cross_entropy(Tensor(z), y).item() == pytest.approx(naive)
    assert softmax_cross_entropy(Tensor(z + 1000.0), y).item() == pytest.approx(naive)
