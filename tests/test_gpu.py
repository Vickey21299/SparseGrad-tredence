"""The GPU (CuPy) path must compute the same thing as the NumPy reference.

Skipped automatically where CuPy or a CUDA device is unavailable.
"""
import numpy as np
import pytest

from engine import Tensor, backend, softmax_cross_entropy
from nn import Adam, Linear, mlp

cp = backend.cp
try:
    gpu_available = cp is not None and cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # driver missing or broken
    gpu_available = False

pytestmark = pytest.mark.skipif(not gpu_available, reason="needs CuPy and a CUDA GPU")


@pytest.fixture
def on_gpu():
    backend.set_device("gpu")
    yield
    backend.set_device("cpu")


def _train_steps(device, steps=5):
    backend.set_device(device)
    rng = np.random.default_rng(0)
    model = mlp([20, 16, 4], rng, dtype=np.float64)
    weights = [m.weight for m in model.modules() if isinstance(m, Linear)]
    weights[0].set_mask(rng.random(weights[0].shape) > 0.5)
    X, y = rng.normal(size=(32, 20)), rng.integers(0, 4, size=32)
    opt = Adam(model.parameters(), lr=1e-2)
    losses = []
    for step in range(steps):
        if step == 3:  # prune more mid-run so the reset path runs on the device too
            weights[1].set_mask(rng.random(weights[1].shape) > 0.3)
        opt.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(X)), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    state = model.state_dict()
    backend.set_device("cpu")
    return losses, state, [backend.to_numpy(w.mask) for w in weights]


def test_gpu_training_matches_cpu_reference():
    cpu_losses, cpu_state, cpu_masks = _train_steps("cpu")
    gpu_losses, gpu_state, gpu_masks = _train_steps("gpu")
    np.testing.assert_allclose(gpu_losses, cpu_losses, rtol=1e-10)
    for key in cpu_state:
        np.testing.assert_allclose(gpu_state[key], cpu_state[key], rtol=1e-9, atol=1e-12)
    for c, g in zip(cpu_masks, gpu_masks):
        np.testing.assert_array_equal(c, g)
    # Pruned weights are exact zeros on the device as well.
    assert np.all(gpu_state["p0"][~gpu_masks[0]] == 0.0)


def test_tensors_are_created_on_the_selected_device(on_gpu):
    assert isinstance(Tensor(np.ones((2, 3))).data, cp.ndarray)
    w = Tensor(np.ones((2, 3)), requires_grad=True)
    (w * 3.0).sum().backward()
    assert isinstance(w.grad, cp.ndarray)
    np.testing.assert_allclose(backend.to_numpy(w.grad), 3.0)
