"""Array backend: NumPy on the CPU (default), or CuPy on an NVIDIA GPU.

The engine is written against the NumPy API. CuPy implements the same API on
CUDA, so every op picks its module from the arrays it receives (`xp_of`) and
the same code runs on either device. NumPy remains the reference
implementation: the tests and the committed results use the CPU path. The GPU
path only changes where the identical arithmetic runs.
"""
import warnings

import numpy as np

try:
    with warnings.catch_warnings():
        # CUDA libraries installed from pip (cupy-cuda12x[ctk]) need no CUDA_PATH.
        warnings.simplefilter("ignore", UserWarning)
        import cupy as cp
except ImportError:  # CPU-only install
    cp = None

_device = "cpu"


def set_device(name):
    """Select where new tensors are created: "cpu" or "gpu"."""
    global _device
    if name not in ("cpu", "gpu"):
        raise ValueError(f"unknown device {name!r}")
    if name == "gpu":
        if cp is None:
            raise RuntimeError("--device gpu needs CuPy: uv pip install 'cupy-cuda12x[ctk]'")
        cp.cuda.runtime.getDeviceCount()  # fails early if no CUDA device/driver
    _device = name


def device():
    return _device


def xp():
    """The array module for the current device."""
    return cp if _device == "gpu" else np


def xp_of(*arrays):
    """The array module that owns `arrays` (NumPy unless one of them is a CuPy array)."""
    if cp is not None:
        return cp.get_array_module(*arrays)
    return np


def is_gpu_array(array):
    return cp is not None and isinstance(array, cp.ndarray)


def to_device(array, dtype=None):
    """Move/convert to an array on the current device."""
    return xp().asarray(array, dtype=dtype)


def to_numpy(array):
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def device_name():
    if _device == "gpu":
        return cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
    return "cpu (NumPy)"
