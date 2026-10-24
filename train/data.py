"""MNIST loading and mini-batching.

scikit-learn is used ONLY to download MNIST (fetch_openml); everything after
loading is NumPy.
"""
from pathlib import Path

import numpy as np

from engine import backend

N_TRAIN = 60_000  # OpenML mnist_784 keeps the standard 60k train / 10k test order


def _load_raw(data_home):
    cache = Path(data_home) / "mnist.npz"
    if cache.exists():
        with np.load(cache) as f:
            return f["X"], f["y"]
    from sklearn.datasets import fetch_openml  # dataset download only

    # parser="liac-arff" avoids needing pandas.
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                        return_X_y=True, data_home=str(data_home))
    X, y = X.astype(np.uint8), y.astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)  # ARFF parsing takes ~15 s; the .npz takes <1 s
    return X, y


def load_mnist(data_home="data", val_size=5000, split_seed=0, dtype=np.float32):
    """Train / validation / test splits, standardized with training-set statistics.

    The validation set is carved out of the 60k training images with a fixed
    `split_seed`, so every run sees the same split; the 10k test set is only
    used for the final report.
    """
    X, y = _load_raw(data_home)
    X = X.astype(dtype) / 255.0
    X_full, y_full, X_test, y_test = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]

    order = np.random.default_rng(split_seed).permutation(N_TRAIN)
    val_idx, train_idx = order[:val_size], order[val_size:]
    X_train, y_train = X_full[train_idx], y_full[train_idx]

    mean, std = X_train.mean(), X_train.std()
    standardize = lambda a: ((a - mean) / std).astype(dtype)
    return {
        "train": (standardize(X_train), y_train),
        "val": (standardize(X_full[val_idx]), y_full[val_idx]),
        "test": (standardize(X_test), y_test),
    }


def to_device(data):
    """Copy every split to the current device once, so batches never cross the PCIe bus."""
    return {name: (backend.to_device(X), backend.to_device(y)) for name, (X, y) in data.items()}


def iterate_minibatches(X, y, batch_size, rng):
    """One shuffled pass over (X, y); the last batch may be smaller.

    The shuffle always comes from the NumPy `rng`, so CPU and GPU runs see the
    same batches in the same order.
    """
    order = backend.xp_of(X).asarray(rng.permutation(len(X)))
    for start in range(0, len(X), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]
