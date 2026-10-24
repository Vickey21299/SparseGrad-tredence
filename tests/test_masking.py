"""Part 1 correctness requirement: masked weights, their gradients, and optimizer state.

Claims under test:
  1. A pruned weight is a real zero and contributes nothing to the forward pass.
  2. dL/dW is exactly 0 on pruned entries (chain rule through W * M) and matches
     finite differences everywhere, for whatever mask is current.
  3. The dense gradient dL/dW_eff is kept separately and is correct on pruned entries.
  4. Training never moves a pruned weight off exactly 0.
  5. A pruned-then-revived connection behaves exactly like a brand-new one,
     and connections that were never pruned are unaffected.
  6. Textbook Adam fails 4 and 5, so these tests can actually detect the bug.
"""
import numpy as np
import pytest

from engine import Parameter, Tensor, softmax_cross_entropy
from engine.gradcheck import check_gradients, numerical_grad
from nn import SGD, Adam


def _problem(seed=0, n=8, d_in=5, d_out=3):
    rng = np.random.default_rng(seed)
    x = Tensor(rng.normal(size=(n, d_in)))
    y = rng.integers(0, d_out, size=n)
    w = Parameter(rng.normal(size=(d_in, d_out)))
    mask = rng.random((d_in, d_out)) > 0.5
    return x, y, w, mask


def _loss(x, y, w):
    return softmax_cross_entropy((x @ w.masked()).tanh(), y)


def test_pruned_weights_are_exact_zeros():
    _, _, w, mask = _problem()
    w.set_mask(mask)
    assert np.all(w.data[~mask] == 0.0)
    assert np.all(w.data[mask] != 0.0)


def test_masked_weight_contributes_nothing_to_forward():
    x, _, w, mask = _problem()
    w.set_mask(mask)
    reference = (x @ w.masked()).data.copy()
    # Even if something wrote garbage into a pruned slot, the forward pass ignores it.
    w.data[~mask] = 1e6
    np.testing.assert_array_equal((x @ w.masked()).data, reference)
    np.testing.assert_array_equal(reference, x.data @ np.where(mask, w.data, 0.0))


def test_gradient_of_masked_weight_is_exactly_zero_and_correct():
    x, y, w, mask = _problem()
    w.set_mask(mask)
    _loss(x, y, w).backward()
    assert np.all(w.grad[~mask] == 0.0)
    assert np.any(w.grad[mask] != 0.0)
    check_gradients(lambda: _loss(x, y, w), [w])


def test_dense_grad_is_gradient_with_respect_to_effective_weight():
    x, y, w, mask = _problem()
    w.set_mask(mask)
    w.zero_grad()
    _loss(x, y, w).backward()

    effective = Tensor(w.data * mask, requires_grad=True)
    expected = numerical_grad(lambda: softmax_cross_entropy((x @ effective).tanh(), y), effective)
    np.testing.assert_allclose(w.dense_grad, expected, rtol=1e-5, atol=1e-8)
    # The signal a regrowth criterion needs is non-zero exactly where grad is zero.
    assert np.any(w.dense_grad[~mask] != 0.0)


def test_gradients_stay_correct_as_mask_changes_between_steps():
    x, y, w, _ = _problem()
    rng = np.random.default_rng(42)
    original = w.data.copy()
    for density in (0.9, 0.5, 0.2, 0.7, 1.0):
        w.data[:] = original  # revive everything at its old value, then re-prune
        mask = rng.random(w.shape) < density
        w.set_mask(mask)
        check_gradients(lambda: _loss(x, y, w), [w])
        assert np.all(w.grad[~mask] == 0.0)


OPTIMIZERS = [
    pytest.param(lambda ps: Adam(ps, lr=0.05), id="adam"),
    pytest.param(lambda ps: SGD(ps, lr=0.1, momentum=0.9), id="sgd-momentum"),
]


@pytest.mark.parametrize("make_optimizer", OPTIMIZERS)
def test_pruned_weights_stay_exactly_zero_while_training(make_optimizer):
    x, y, w, _ = _problem()
    opt = make_optimizer([w])
    losses = []
    for step in range(60):
        if step == 20:
            # Prune mid-training, when the optimizer already holds momentum
            # for every entry. That momentum must not leak into pruned slots.
            w.set_mask(np.abs(w.data) > np.median(np.abs(w.data)))
        opt.zero_grad()
        loss = _loss(x, y, w)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if w.mask is not None:
            assert np.all(w.data[~w.mask] == 0.0), f"pruned weight leaked at step {step}"
    assert losses[-1] < losses[20]


def test_optimizer_ignores_gradient_on_pruned_entries_even_if_masking_was_bypassed():
    w = Parameter(np.ones(4))
    w.set_mask([True, False, True, False])
    opt = Adam([w], lr=0.1)
    w.grad = np.full(4, 5.0)  # e.g. a caller used `w` instead of `w.masked()`
    opt.step()
    np.testing.assert_array_equal(w.data[[1, 3]], 0.0)


# ---- revival -----------------------------------------------------------

def _gradient_stream(steps, size, seed=7):
    return np.random.default_rng(seed).normal(size=(steps, size))


def _run(opt_cls, kwargs, grads, prune_at=None, revive_at=None, entry=0):
    """Feed fixed gradients to a parameter, pruning and reviving one entry."""
    w = Parameter(np.full(grads.shape[1], 0.5))
    opt = opt_cls([w], **kwargs)
    trajectory = []
    for step, g in enumerate(grads):
        if step == prune_at:
            mask = np.ones(w.shape, dtype=bool)
            mask[entry] = False
            w.set_mask(mask)
        if step == revive_at:
            w.set_mask(np.ones(w.shape, dtype=bool))
        w.grad = g.copy()
        opt.step()
        trajectory.append(w.data.copy())
    return np.array(trajectory)


@pytest.mark.parametrize(
    "opt_cls, kwargs",
    [(Adam, {"lr": 0.01}), (SGD, {"lr": 0.01, "momentum": 0.9})],
    ids=["adam", "sgd-momentum"],
)
def test_revived_connection_behaves_like_a_brand_new_connection(opt_cls, kwargs):
    grads = _gradient_stream(30, 3)
    prune_at, revive_at = 10, 20
    trajectory = _run(opt_cls, kwargs, grads, prune_at, revive_at)

    # Reference: a fresh optimizer on a fresh weight that starts at 0 at the
    # revival step and sees the same gradients from then on.
    fresh = Parameter(np.zeros(1))
    fresh_opt = opt_cls([fresh], **kwargs)
    reference = []
    for g in grads[revive_at:, :1]:
        fresh.grad = g.copy()
        fresh_opt.step()
        reference.append(fresh.data[0])

    np.testing.assert_array_equal(trajectory[prune_at:revive_at, 0], 0.0)
    np.testing.assert_allclose(trajectory[revive_at:, 0], reference, rtol=1e-12, atol=0)

    # Entries that were never pruned follow exactly the trajectory of a run without pruning.
    unpruned = _run(opt_cls, kwargs, grads)
    np.testing.assert_allclose(trajectory[:, 1:], unpruned[:, 1:], rtol=1e-12, atol=0)


def test_revival_before_next_step_still_resets_state():
    w = Parameter(np.full(2, 0.5))
    opt = Adam([w], lr=0.01)
    for g in _gradient_stream(5, 2):
        w.grad = g
        opt.step()
    w.set_mask([False, True])  # prune ...
    w.set_mask([True, True])   # ... and revive before the optimizer runs again
    w.grad = np.array([0.3, 0.3])
    opt.step()
    # A fresh Adam's first step is lr * sign(g) (up to eps).
    assert w.data[0] == pytest.approx(-0.01, rel=1e-6)


# ---- the bug these tests exist to catch ---------------------------------

class TextbookAdam:
    """Standard Adam with a global step count and no knowledge of masks."""

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, (self.b1, self.b2), self.eps = list(params), lr, betas, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * p.grad
            v[:] = self.b2 * v + (1 - self.b2) * p.grad**2
            m_hat, v_hat = m / (1 - self.b1**self.t), v / (1 - self.b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def test_textbook_adam_leaks_pruned_weights_even_with_zero_gradient():
    grads = _gradient_stream(30, 3)
    w = Parameter(np.full(3, 0.5))
    opt = TextbookAdam([w], lr=0.01)
    for step, g in enumerate(grads):
        if step == 10:
            w.set_mask([False, True, True])
        w.grad = g.copy()
        if w.mask is not None:
            w.grad[~w.mask] = 0.0  # the engine really does deliver an exact zero
        opt.step()
    # Zero gradient, yet stale momentum pushed the "pruned" weight away from 0.
    assert w.data[0] != 0.0


def test_textbook_adam_overshoots_first_step_after_revival():
    grads = _gradient_stream(6, 1)
    w = Parameter(np.full(1, 0.5))
    opt = TextbookAdam([w], lr=0.01)
    for g in grads[:-1]:
        w.grad = g.copy()
        opt.step()
    opt.t = 10_000  # deep into training, where b1**t and b2**t are ~0
    # Even if we reset the revived connection's moments and weight by hand,
    # bias correction still uses the global step (~1 for large t), so the
    # first step is (1 - b1) / sqrt(1 - b2) ~= 3.16x the fresh-Adam step.
    opt.m[0][:] = 0.0
    opt.v[0][:] = 0.0
    w.data[:] = 0.0
    w.grad = grads[-1].copy()
    opt.step()
    assert abs(w.data[0]) == pytest.approx(0.01 * 0.1 / np.sqrt(0.001), rel=0.05)
    assert abs(w.data[0]) > 3 * 0.01
