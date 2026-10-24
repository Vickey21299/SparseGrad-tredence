"""Mask-aware optimizers.

A pruned entry is treated as a connection that does not exist:
  * its gradient is ignored, even if a caller bypassed `Parameter.masked()`;
  * its optimizer state (momentum / Adam moments / step count) is held at
    zero, so stale momentum cannot drag a pruned weight off zero;
  * when it is revived it starts exactly like a freshly created connection:
    weight 0, empty state, and Adam bias correction counted from its own first
    step, not from the global step. With a global step count, a revived
    entry's first update would be (1-b1)/sqrt(1-b2) ~= 3.16x too large.
"""
from engine import backend


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    @staticmethod
    def _active_grad(p):
        if p.mask is None:
            return p.grad
        return backend.xp_of(p.grad).where(p.mask, p.grad, p.grad.dtype.type(0))

    @staticmethod
    def _entries_to_reset(p):
        """Entries whose state must be cleared: pruned now, or pruned since the last step."""
        if p.mask is None:
            return None
        reset = ~p.mask
        if p.pruned_since_step is not None:
            reset |= p.pruned_since_step
            p.pruned_since_step = None
        return reset


class SGD(Optimizer):
    """SGD with (heavy-ball) momentum: v <- mu v + g,  w <- w - lr v."""

    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [backend.xp_of(p.data).zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            reset = self._entries_to_reset(p)
            if reset is not None:
                v[reset] = 0.0
            v *= self.momentum
            v += self._active_grad(p)
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam with per-entry bias correction, so revived connections restart cleanly.

    Each entry i has its own step count t_i. Instead of storing t_i and
    computing beta**t_i every step, beta1**t_i and beta2**t_i are kept directly
    (`pow1`, `pow2`) and updated with one multiply. A reset entry goes back to
    beta**0 = 1, which starts its own clock again.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        zeros = lambda p: backend.xp_of(p.data).zeros_like(p.data)
        ones = lambda p: backend.xp_of(p.data).ones_like(p.data)
        self.m = [zeros(p) for p in self.params]
        self.v = [zeros(p) for p in self.params]
        self.pow1 = [ones(p) for p in self.params]
        self.pow2 = [ones(p) for p in self.params]

    def step(self):
        b1, b2 = self.beta1, self.beta2
        for p, m, v, pow1, pow2 in zip(self.params, self.m, self.v, self.pow1, self.pow2):
            if p.grad is None:
                continue
            if backend.is_gpu_array(p.data):
                self._fused_gpu_step(p, m, v, pow1, pow2)
                continue
            g = self._active_grad(p)
            reset = self._entries_to_reset(p)
            if reset is None:
                pow1 *= b1
                pow2 *= b2
            else:
                m[reset] = 0.0
                v[reset] = 0.0
                pow1[reset] = 1.0
                pow2[reset] = 1.0
                # Only connections that exist take a step: factor beta if active, 1 if pruned.
                active = p._mask_float
                pow1 *= 1.0 - (1.0 - b1) * active
                pow2 *= 1.0 - (1.0 - b2) * active
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            # Active entries have t >= 1, so 1 - beta**t >= 1 - beta and the max
            # changes nothing. Pruned entries (beta**0 = 1) would give 0/0; they
            # get 1 - beta instead, and since m = 0 their update is exactly 0.
            xp = backend.xp_of(p.data)
            m_hat = m / xp.maximum(1.0 - pow1, 1.0 - b1)
            v_hat = v / xp.maximum(1.0 - pow2, 1.0 - b2)
            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def _fused_gpu_step(self, p, m, v, pow1, pow2):
        """The same update as the NumPy path above, as ONE GPU kernel.

        Element-wise, the NumPy path launches ~15 separate kernels per parameter.
        For a model this small, kernel launch overhead dominates the GPU time.
        tests/test_gpu.py checks that both paths agree.
        """
        reset = self._entries_to_reset(p)
        active = p._mask_float if p.mask is not None else 1.0
        reset = reset if reset is not None else False
        _adam_kernel()(p.grad, active, reset, p.dtype.type(self.lr), p.dtype.type(self.beta1),
                       p.dtype.type(self.beta2), p.dtype.type(self.eps), p.data, m, v, pow1, pow2)


_ADAM_KERNEL = None


def _adam_kernel():
    global _ADAM_KERNEL
    if _ADAM_KERNEL is None:
        _ADAM_KERNEL = backend.cp.ElementwiseKernel(
            "T g, T active, bool reset, T lr, T b1, T b2, T eps",
            "T p, T m, T v, T pow1, T pow2",
            """
            if (reset) { m = 0; v = 0; pow1 = 1; pow2 = 1; }
            T ga = active != 0 ? g : (T)0;              // pruned: ignore the gradient
            if (active != 0) { pow1 *= b1; pow2 *= b2; } // pruned: per-entry clock stays at 0
            m = b1 * m + (1 - b1) * ga;
            v = b2 * v + (1 - b2) * ga * ga;
            T m_hat = m / max(1 - pow1, 1 - b1);
            T v_hat = v / max(1 - pow2, 1 - b2);
            p -= lr * m_hat / (sqrt(v_hat) + eps);
            """,
            "masked_adam",
        )
    return _ADAM_KERNEL
