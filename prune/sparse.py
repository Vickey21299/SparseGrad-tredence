"""Sparse-aware inference: compute with the surviving weights only.

Training uses the honest-but-dense shortcut x @ (W * M): the zeros are real,
but a dense matmul still multiplies them, so it does the same work at any
sparsity. These two forward paths actually skip the pruned weights:

  CSRLinear   unstructured: one multiply-add per surviving weight, O(batch x nnz).
  compact_mlp structural: pruning often kills whole units (an input pixel whose
              every outgoing weight was pruned, a hidden neuron with no inputs or
              no outputs). Removing them yields smaller *dense* matrices that run
              on BLAS at full speed.

Both are checked against the dense masked forward in tests/test_sparse.py.
"""
import numpy as np


class CSRLinear:
    """y = x @ W + b, with W stored per output unit as (input indices, values) in CSR form."""

    def __init__(self, W, b):
        W = np.asarray(W, dtype=np.float32)
        self.in_features, self.out_features = W.shape
        out_idx, in_idx = np.nonzero(W.T)  # row-major over W.T, so grouped by output unit
        self.indices = in_idx.astype(np.int32)
        self.values = W.T[out_idx, in_idx]
        counts = np.bincount(out_idx, minlength=self.out_features)
        self.indptr = np.concatenate([[0], np.cumsum(counts)])
        self.nonempty = np.flatnonzero(counts)       # outputs with at least one input
        self.starts = self.indptr[:-1][self.nonempty]
        self.bias = np.asarray(b, dtype=np.float32)

    @property
    def nnz(self):
        return len(self.values)

    def nbytes(self):
        return self.values.nbytes + self.indices.nbytes + self.indptr.astype(np.int32).nbytes + self.bias.nbytes

    def __call__(self, x):
        out = np.empty((x.shape[0], self.out_features), dtype=np.float32)
        out[:] = self.bias  # outputs with no surviving inputs are just the bias
        if self.nnz == 0:
            return out
        # Each surviving weight costs one gather + one multiply per example; the
        # per-output sums are segment sums. Chunk the batch so the (B, nnz)
        # product buffer stays ~16 MB.
        chunk = max(1, (1 << 22) // self.nnz)
        for s in range(0, len(x), chunk):
            products = x[s:s + chunk, self.indices] * self.values
            out[s:s + chunk, self.nonempty] += np.add.reduceat(products, self.starts, axis=1)
        return out


class CSRMLP:
    def __init__(self, layers):
        self.layers = [CSRLinear(W, b) for W, b in layers]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                np.maximum(x, 0, out=x)
        return x

    def nbytes(self):
        return sum(layer.nbytes() for layer in self.layers)


class CompactMLP:
    """A dense ReLU MLP on a subset of the input features."""

    def __init__(self, input_index, layers):
        self.input_index = input_index
        self.layers = [(np.ascontiguousarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32))
                       for W, b in layers]

    def __call__(self, x):
        x = x[:, self.input_index]
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                np.maximum(x, 0, out=x)
        return x

    @property
    def shapes(self):
        return [W.shape for W, _ in self.layers]

    def nbytes(self):
        return self.input_index.nbytes + sum(W.nbytes + b.nbytes for W, b in self.layers)


def compact_mlp(layers):
    """Remove every unit that provably cannot change the output, without changing the function.

    - input feature with no outgoing weights   -> drop it (never read)
    - hidden unit with no outgoing weights     -> drop it
    - hidden unit with no incoming weights     -> its output is the constant relu(b_j);
                                                  fold relu(b_j) * W_next[j] into the next
                                                  bias, then drop it
    Repeats until nothing changes (removing a unit can orphan another).
    """
    layers = [(np.array(W, dtype=np.float64), np.array(b, dtype=np.float64)) for W, b in layers]
    input_index = np.flatnonzero(np.any(layers[0][0] != 0, axis=1))
    layers[0] = (layers[0][0][input_index], layers[0][1])

    changed = True
    while changed:
        changed = False
        for i in range(len(layers) - 1):
            (W, b), (W_next, b_next) = layers[i], layers[i + 1]
            no_out = ~np.any(W_next != 0, axis=1)
            no_in = ~np.any(W != 0, axis=0)
            b_next = b_next + np.maximum(b[no_in & ~no_out], 0) @ W_next[no_in & ~no_out]
            keep = ~(no_out | no_in)
            if not keep.all():
                changed = True
            layers[i] = (W[:, keep], b[keep])
            layers[i + 1] = (W_next[keep], b_next)
    return CompactMLP(input_index.astype(np.int64), layers)


def dense_forward(layers, x):
    """Reference: the dense forward pass (dense-times-zero for pruned weights)."""
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            np.maximum(x, 0, out=x)
    return x
