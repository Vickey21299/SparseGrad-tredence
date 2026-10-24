"""Finite-difference gradient checking for the autodiff engine."""
import numpy as np


def numerical_grad(f, x, eps=1e-6):
    """Central-difference d f() / d x for a scalar-valued closure f.

    `x.data` is perturbed in place one entry at a time and restored afterwards.
    """
    grad = np.zeros_like(x.data)
    for i in np.ndindex(x.data.shape):
        original = x.data[i]
        x.data[i] = original + eps
        plus = float(f().data)
        x.data[i] = original - eps
        minus = float(f().data)
        x.data[i] = original
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def check_gradients(f, inputs, eps=1e-6, rtol=1e-5, atol=1e-8):
    """Assert that backprop through f() matches finite differences for every input."""
    for x in inputs:
        x.zero_grad()
    f().backward()
    analytic = [np.zeros_like(x.data) if x.grad is None else x.grad.copy() for x in inputs]
    for index, (x, expected) in enumerate(zip(inputs, analytic)):
        np.testing.assert_allclose(
            expected, numerical_grad(f, x, eps), rtol=rtol, atol=atol,
            err_msg=f"gradient mismatch for input {index} with shape {x.shape}",
        )
