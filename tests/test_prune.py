"""Part 3: schedule, criterion quality, exact sparsity, honest zeros, regrowth."""
import numpy as np
import pytest

from engine import Parameter, Tensor, softmax_cross_entropy
from nn import Adam, Linear, mlp
from prune import Pruner, cosine_decay, cubic_sparsity, importance


def test_cubic_schedule_endpoints_monotone_and_front_loaded():
    values = [cubic_sparsity(t, 100, 1100, 0.9) for t in range(0, 1300, 10)]
    assert values[0] == 0.0 and values[-1] == pytest.approx(0.9)
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert cubic_sparsity(600, 100, 1100, 0.9) == pytest.approx(0.9 * 7 / 8)


def test_cosine_decay_reaches_zero():
    assert cosine_decay(0, 0, 100, 0.3) == 0.3
    assert cosine_decay(50, 0, 100, 0.3) == pytest.approx(0.15)
    assert cosine_decay(100, 0, 100, 0.3) == 0.0


def _rank_correlation(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def test_taylor_criterion_predicts_true_loss_increase_better_than_magnitude():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 8))
    y = (X[:, :3].sum(axis=1) > 0).astype(int) + (X[:, 3] > 1)
    model = mlp([8, 16, 3], rng, dtype=np.float64)
    opt = Adam(model.parameters(), lr=1e-2)
    loss_fn = lambda: softmax_cross_entropy(model(Tensor(X)), y)
    for _ in range(150):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()

    w = model.layers[0].weight
    base = loss_fn().item()
    # Gradient statistics from mini-batches of 16, exactly as the Pruner collects them.
    batch, grads = 16, []
    for start in range(0, len(X), batch):
        w.zero_grad()
        softmax_cross_entropy(model(Tensor(X[start:start + batch])), y[start:start + batch]).backward()
        grads.append(w.grad.copy())
    grads = np.array(grads)
    g_mean, g2_mean = grads.mean(axis=0), (grads ** 2).mean(axis=0)

    true_increase = np.zeros(w.shape)
    for i in np.ndindex(w.shape):
        saved = w.data[i]
        w.data[i] = 0.0
        true_increase[i] = loss_fn().item() - base
        w.data[i] = saved

    corr = {c: _rank_correlation(importance(c, w.data, g_mean, g2_mean, batch).ravel(), true_increase.ravel())
            for c in ("magnitude", "taylor1", "fisher", "taylor2")}
    # The default criterion must beat the magnitude baseline at predicting the real damage.
    assert corr["fisher"] > corr["magnitude"], corr
    assert corr["fisher"] > 0.9, corr


def _tiny_model(seed=0):
    rng = np.random.default_rng(seed)
    model = mlp([20, 30, 10, 4], rng, dtype=np.float64)
    X, y = rng.normal(size=(64, 20)), rng.integers(0, 4, size=64)
    return model, [m.weight for m in model.modules() if isinstance(m, Linear)], X, y


def _run(model, weights, X, y, pruner, steps):
    opt = Adam(model.parameters(), lr=1e-2)
    for step in range(1, steps + 1):
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X)), y).backward()
        opt.step()
        pruner(step)
        for w in weights:  # honest masking holds after every single step
            assert np.all(w.data[~w.mask] == 0.0)


@pytest.mark.parametrize("criterion", ["magnitude", "taylor1", "fisher", "taylor2"])
@pytest.mark.parametrize("regrow", [0.0, 0.3])
def test_pruner_hits_exact_target_with_real_zeros(criterion, regrow):
    model, weights, X, y = _tiny_model()
    pruner = Pruner(weights, 0.9, start_step=5, end_step=45, every=5, batch_size=64,
                    criterion=criterion, regrow=regrow)
    _run(model, weights, X, y, pruner, steps=60)
    total = sum(w.data.size for w in weights)
    assert sum(int(w.mask.sum()) for w in weights) == total - round(0.9 * total)
    assert pruner.exact_zero_fraction() >= 0.9
    targets = [h["target_sparsity"] for h in pruner.history]
    assert targets == sorted(targets) and targets[-1] == pytest.approx(0.9)


def test_every_layer_keeps_its_floor():
    model, weights, X, y = _tiny_model()
    # 99% of 940 weights leaves a budget of 9, but the 5% floors need 30 + 15 + 2 = 47.
    pruner = Pruner(weights, 0.99, start_step=1, end_step=10, every=1, batch_size=64,
                    criterion="magnitude", min_layer_density=0.05)
    _run(model, weights, X, y, pruner, steps=12)
    for w in weights:
        assert w.mask.sum() >= np.ceil(0.05 * w.data.size)
    assert pruner.sparsity() < 0.99  # the shortfall is visible, not hidden


def test_regrowth_revives_pruned_connections_at_zero():
    w = Parameter(np.linspace(0.1, 1.0, 10).reshape(2, 5))
    pruner = Pruner([w], 0.5, start_step=1, end_step=3, every=1, batch_size=1,
                    criterion="magnitude", regrow=0.0)
    w.dense_grad = np.zeros(w.shape)
    for step in (1, 2, 3):
        pruner(step)
    pruned_before = ~w.mask.copy()
    # A pruned connection now has a huge gradient; with regrowth enabled it must come back.
    # Hold the target at 50% while regrowth is on, by re-running the ramp's last step.
    pruner.regrow, pruner.start_step, pruner.end_step = 0.4, 0, 10
    pruner.final_sparsity = 0.5
    w.dense_grad = np.zeros(w.shape)
    w.dense_grad[0, 0] = 100.0
    assert pruned_before[0, 0]
    pruner(5)
    expected_keep = 10 - round(cubic_sparsity(5, 0, 10, 0.5) * 10)
    assert w.mask[0, 0] and w.data[0, 0] == 0.0
    assert pruner.history[-1]["revived"] >= 1
    assert w.mask.sum() == expected_keep  # regrowth swaps connections; it never exceeds the budget
