"""The sparse inference paths must compute exactly what the masked dense model computes."""
import numpy as np
import pytest

from prune.sparse import CSRLinear, CSRMLP, compact_mlp, dense_forward


def _sparse_mlp(seed, sizes=(30, 20, 12, 4), density=0.15):
    rng = np.random.default_rng(seed)
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(size=(fan_in, fan_out)).astype(np.float32)
        W *= rng.random(W.shape) < density
        layers.append((W, rng.normal(size=fan_out).astype(np.float32)))
    # Force the awkward cases: a dead input, a hidden unit with no inputs (but
    # outgoing weights and a positive bias), and one with no outputs.
    layers[0][0][3, :] = 0
    layers[0][0][:, 5] = 0
    layers[0][1][5] = 1.5
    layers[1][0][5, :] = rng.normal(size=sizes[2])
    layers[1][0][7, :] = 0
    return layers


@pytest.mark.parametrize("seed", range(5))
def test_csr_linear_matches_dense(seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(40, 25)).astype(np.float32) * (rng.random((40, 25)) < 0.1)
    W[:, 0] = 0  # an output with no inputs
    b = rng.normal(size=25).astype(np.float32)
    x = rng.normal(size=(17, 40)).astype(np.float32)
    layer = CSRLinear(W, b)
    assert layer.nnz == np.count_nonzero(W)
    np.testing.assert_allclose(layer(x), x @ W + b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("seed", range(5))
def test_csr_and_compact_mlps_match_dense(seed):
    layers = _sparse_mlp(seed)
    x = np.random.default_rng(100 + seed).normal(size=(64, 30)).astype(np.float32)
    expected = dense_forward(layers, x.copy())
    np.testing.assert_allclose(CSRMLP(layers)(x), expected, rtol=1e-4, atol=1e-4)
    compact = compact_mlp(layers)
    np.testing.assert_allclose(compact(x), expected, rtol=1e-4, atol=1e-4)
    # The dead input and both dead hidden units are physically gone.
    assert 3 not in compact.input_index
    assert compact.shapes[0][1] <= 20 - 2


def test_fully_pruned_layer_gives_bias_only():
    layer = CSRLinear(np.zeros((5, 3), dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(layer(np.ones((2, 5), dtype=np.float32)), [[1, 2, 3], [1, 2, 3]])
