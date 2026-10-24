"""Minimal reverse-mode automatic differentiation over NumPy (or CuPy) arrays.

The graph is built dynamically on every forward pass (define-by-run), so a
pruning mask that changes between steps simply produces a different graph on
the next step; nothing about a previous graph is reused.
"""
from __future__ import annotations

import contextlib

import numpy as np

from engine import backend

_GRAD_ENABLED = True


@contextlib.contextmanager
def no_grad():
    """Run the enclosed forward computation without recording a graph."""
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def _as_float_array(data, dtype=None):
    # Arrays already on a device stay there; anything else goes to the current device.
    module = backend.xp_of(data) if hasattr(data, "__cuda_array_interface__") else backend.xp()
    array = module.asarray(data, dtype=dtype)
    if array.dtype.kind != "f":
        array = array.astype(np.float64)
    return array


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting.

    Broadcasting copies an operand along (a) new leading axes and (b) axes of
    size 1. Each copy receives its own gradient, so the operand's gradient is
    the sum over those axes, e.g. the bias in (N, D) + (D,) gets a column sum.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def _accumulate(tensor, grad):
    """Add `grad` into `tensor.grad`; a node used k times receives k contributions."""
    if not tensor.requires_grad:
        return
    grad = _unbroadcast(grad, tensor.data.shape)
    if tensor.grad is None:
        tensor.grad = grad.astype(tensor.data.dtype, copy=True)
    else:
        tensor.grad = tensor.grad + grad


class Tensor:
    """A NumPy array plus the graph edges and gradient needed for reverse-mode AD."""

    # Makes `ndarray <op> Tensor` dispatch to Tensor's reflected operators.
    __array_priority__ = 100

    def __init__(self, data, requires_grad=False, dtype=None):
        self.data = _as_float_array(data, dtype)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = ()
        self._backward = None
        self.op = ""

    @staticmethod
    def _make(data, parents, backward, op):
        out = Tensor(data)
        if _GRAD_ENABLED and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out.op = op
        return out

    def _lift(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(backend.xp_of(self.data).asarray(other, dtype=self.data.dtype))

    # ---- introspection -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data)

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        grad = f", op={self.op!r}" if self.op else ""
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype}{grad})"

    # ---- element-wise arithmetic ---------------------------------------
    def __add__(self, other):
        other = self._lift(other)

        def backward(g):
            _accumulate(self, g)
            _accumulate(other, g)

        return Tensor._make(self.data + other.data, (self, other), backward, "add")

    def __sub__(self, other):
        other = self._lift(other)

        def backward(g):
            _accumulate(self, g)
            _accumulate(other, -g)

        return Tensor._make(self.data - other.data, (self, other), backward, "sub")

    def __mul__(self, other):
        other = self._lift(other)
        a, b = self.data, other.data

        def backward(g):
            _accumulate(self, g * b)
            _accumulate(other, g * a)

        return Tensor._make(a * b, (self, other), backward, "mul")

    def __truediv__(self, other):
        other = self._lift(other)
        a, b = self.data, other.data

        def backward(g):
            _accumulate(self, g / b)
            _accumulate(other, -g * a / (b * b))

        return Tensor._make(a / b, (self, other), backward, "div")

    def __neg__(self):
        def backward(g):
            _accumulate(self, -g)

        return Tensor._make(-self.data, (self,), backward, "neg")

    def __radd__(self, other):
        return self._lift(other) + self

    def __rsub__(self, other):
        return self._lift(other) - self

    def __rmul__(self, other):
        return self._lift(other) * self

    def __rtruediv__(self, other):
        return self._lift(other) / self

    # ---- linear algebra ------------------------------------------------
    def __matmul__(self, other):
        other = self._lift(other)
        a, b = self.data, other.data
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError("matmul expects operands with ndim >= 2")

        def backward(g):
            # C = A B  =>  dA = dC B^T,  dB = A^T dC  (leading batch axes are
            # summed back by _unbroadcast when one side was broadcast).
            if self.requires_grad:
                _accumulate(self, g @ b.swapaxes(-1, -2))
            if other.requires_grad:
                _accumulate(other, a.swapaxes(-1, -2) @ g)

        return Tensor._make(a @ b, (self, other), backward, "matmul")

    # ---- reductions ----------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        if axis is not None:
            axis = tuple(a % self.data.ndim for a in axis)

        xp = backend.xp_of(self.data)

        def backward(g):
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)
            _accumulate(self, xp.broadcast_to(g, self.data.shape))

        out = self.data.sum(axis=axis, keepdims=keepdims)
        return Tensor._make(out, (self,), backward, "sum")

    def mean(self, axis=None, keepdims=False):
        total = self.sum(axis=axis, keepdims=keepdims)
        count = self.data.size // max(total.data.size, 1)
        return total * (1.0 / count)

    # ---- non-linearities -----------------------------------------------
    def relu(self):
        x = self.data

        def backward(g):
            # Sub-gradient 0 at x == 0.
            _accumulate(self, g * (x > 0))

        return Tensor._make(backend.xp_of(x).maximum(x, 0), (self,), backward, "relu")

    def tanh(self):
        t = backend.xp_of(self.data).tanh(self.data)

        def backward(g):
            _accumulate(self, g * (1.0 - t * t))

        return Tensor._make(t, (self,), backward, "tanh")

    def sigmoid(self):
        x = self.data
        xp = backend.xp_of(x)
        # exp is only ever taken of a non-positive number, so it cannot overflow.
        e = xp.exp(-xp.abs(x))
        s = xp.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

        def backward(g):
            _accumulate(self, g * s * (1.0 - s))

        return Tensor._make(s, (self,), backward, "sigmoid")

    # ---- reverse pass --------------------------------------------------
    def backward(self, grad=None):
        """Accumulate d(self)/d(leaf) into every leaf that requires grad."""
        xp = backend.xp_of(self.data)
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on a non-scalar tensor needs an explicit grad")
            grad = xp.ones_like(self.data)

        # Iterative post-order DFS: every node is emitted after all of its
        # inputs, so walking the list in reverse visits a node only once all
        # of its consumers have pushed their gradient into it.
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in visited)

        _accumulate(self, xp.asarray(grad, dtype=self.data.dtype))
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)


class Parameter(Tensor):
    """A trainable leaf tensor with an optional boolean pruning mask.

    The forward pass must use `masked()`, which computes W_eff = W * M.
    By the chain rule dL/dW = dL/dW_eff * M, so `grad` is exactly zero on
    pruned entries and the optimizer never sees a signal there. The full
    dL/dW_eff, which *is* non-zero on pruned entries and estimates how much
    the loss would change if a connection were revived, is kept separately in
    `dense_grad` for pruning/regrowth decisions only.
    """

    def __init__(self, data, dtype=None):
        super().__init__(data, requires_grad=True, dtype=dtype)
        self.mask = None
        self._mask_float = None
        self.dense_grad = None
        # Entries pruned at any point since the optimizer last stepped. Lets the
        # optimizer reset their state even if they were revived before it ran.
        self.pruned_since_step = None

    def set_mask(self, mask):
        mask = backend.xp_of(self.data).asarray(mask, dtype=bool)
        if mask.shape != self.data.shape:
            raise ValueError(f"mask shape {mask.shape} != parameter shape {self.data.shape}")
        self.mask = mask
        self._mask_float = mask.astype(self.data.dtype)  # reused by every forward pass
        if self.pruned_since_step is None:
            self.pruned_since_step = ~mask
        else:
            self.pruned_since_step |= ~mask
        # A pruned connection is a real zero, not a small number. Revived
        # entries were zeroed when pruned, so they restart from 0.
        self.data[~mask] = 0.0

    @property
    def density(self):
        return 1.0 if self.mask is None else float(self.mask.mean())

    def masked(self):
        """The effective weight W * M to use in the forward pass."""
        if self.mask is None:
            return self
        m = self._mask_float

        def backward(g):
            _accumulate(self, g * m)
            self.dense_grad = g.copy() if self.dense_grad is None else self.dense_grad + g

        return Tensor._make(self.data * m, (self,), backward, "mask")

    def zero_grad(self):
        self.grad = None
        self.dense_grad = None

    def __repr__(self):
        return f"Parameter(shape={self.data.shape}, density={self.density:.3f})"


def softmax_cross_entropy(logits, targets):
    """Mean cross-entropy of softmax(logits) against integer class labels.

    Fused for stability: log_softmax = z - max(z) - log(sum(exp(z - max(z)))),
    which never overflows, and the gradient is the well-conditioned
    (softmax(z) - onehot(y)) / N instead of chaining through a division by a
    probability that may have underflowed to 0.
    """
    z = logits.data
    if z.ndim != 2:
        raise ValueError("logits must have shape (N, C)")
    xp = backend.xp_of(z)
    n = z.shape[0]
    rows = xp.arange(n)
    targets = xp.asarray(targets, dtype=np.int64)
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - xp.log(xp.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[rows, targets].mean()

    def backward(g):
        grad = xp.exp(log_probs)
        grad[rows, targets] -= 1.0
        _accumulate(logits, g * grad / n)

    return Tensor._make(xp.asarray(loss, dtype=z.dtype), (logits,), backward, "softmax_ce")
