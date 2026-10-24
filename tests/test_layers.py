"""Part 2 building blocks: layers, init, data batching, and a tiny end-to-end fit."""
import numpy as np

from engine import Tensor, softmax_cross_entropy
from engine.gradcheck import check_gradients
from nn import Adam, Linear, mlp
from train.data import iterate_minibatches


def test_linear_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng, dtype=np.float64)
    layer.bias.data[:] = rng.normal(size=3)
    x = Tensor(rng.normal(size=(5, 4)), requires_grad=True)
    y = np.array([0, 1, 2, 1, 0])
    check_gradients(lambda: softmax_cross_entropy(layer(x), y), [x, layer.weight, layer.bias])


def test_linear_uses_masked_weight():
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng, dtype=np.float64)
    mask = rng.random((4, 3)) > 0.5
    layer.weight.set_mask(mask)
    x = rng.normal(size=(2, 4))
    np.testing.assert_allclose(layer(Tensor(x)).data, x @ (layer.weight.data * mask))


def test_mlp_collects_all_parameters():
    model = mlp([8, 6, 4, 3], np.random.default_rng(0))
    shapes = [p.shape for p in model.parameters()]
    assert shapes == [(8, 6), (6,), (6, 4), (4,), (4, 3), (3,)]


def test_he_init_preserves_activation_scale_through_relu_layers():
    rng = np.random.default_rng(0)
    model = mlp([512] * 6 + [10], rng)
    h = Tensor(rng.normal(size=(2000, 512)).astype(np.float32))
    stds = []
    for layer in model.layers:
        h = layer(h)
        if not isinstance(layer, Linear):
            stds.append(float(np.sqrt((h.data ** 2).mean())))
    # Second moment after ReLU stays ~1 at every depth (no vanishing / exploding).
    assert all(0.8 < s < 1.25 for s in stds), stds


def test_minibatches_cover_every_example_once():
    X, y = np.arange(10)[:, None].astype(float), np.arange(10)
    seen = np.concatenate([yb for _, yb in iterate_minibatches(X, y, 3, np.random.default_rng(0))])
    assert sorted(seen.tolist()) == list(range(10))


def test_mlp_can_overfit_a_tiny_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 10)).astype(np.float32)
    y = rng.integers(0, 3, size=32)
    model = mlp([10, 32, 3], rng)
    opt = Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(X)), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert (model(Tensor(X)).data.argmax(axis=1) == y).all()
