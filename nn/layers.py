"""Composable layers built on the Part 1 engine."""
import numpy as np

from engine import Parameter, backend


class Module:
    """Base class: finds Parameters and sub-Modules stored as attributes or in lists."""

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def children(self):
        for value in vars(self).values():
            items = value if isinstance(value, (list, tuple)) else (value,)
            for item in items:
                if isinstance(item, Module):
                    yield item

    def modules(self):
        yield self
        for child in self.children():
            yield from child.modules()

    def parameters(self):
        params = []
        for module in self.modules():
            params.extend(v for v in vars(module).values() if isinstance(v, Parameter))
        return params

    def state_dict(self):
        """Parameter values as NumPy arrays (copied off the GPU if needed)."""
        return {f"p{i}": backend.to_numpy(p.data).copy() for i, p in enumerate(self.parameters())}

    def load_state_dict(self, state):
        for i, p in enumerate(self.parameters()):
            p.data[...] = backend.xp_of(p.data).asarray(state[f"p{i}"])


def init_std(scheme, fan_in, fan_out):
    """Standard deviation of the initial weights.

    he:     Var(w) = 2 / fan_in. A ReLU zeroes half its inputs, so this keeps the
            variance of pre-activations constant from layer to layer.
    xavier: Var(w) = 2 / (fan_in + fan_out), for tanh/sigmoid, which are roughly
            linear near 0 and need both forward and backward variance preserved.
    float:  a fixed std, used only to show what goes wrong (vanishing/exploding).
    """
    if scheme == "he":
        return np.sqrt(2.0 / fan_in)
    if scheme == "xavier":
        return np.sqrt(2.0 / (fan_in + fan_out))
    return float(scheme)


class Linear(Module):
    """y = x @ (W * M) + b. Only the weight matrix is prunable; biases stay dense."""

    def __init__(self, in_features, out_features, rng, init="he", dtype=np.float32):
        std = init_std(init, in_features, out_features)
        self.weight = Parameter(rng.normal(0.0, std, size=(in_features, out_features)).astype(dtype))
        self.bias = Parameter(np.zeros(out_features, dtype=dtype))

    def forward(self, x):
        return x @ self.weight.masked() + self.bias


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


ACTIVATIONS = {"relu": (ReLU, "he"), "tanh": (Tanh, "xavier")}


def mlp(sizes, rng, activation="relu", init=None, dtype=np.float32):
    """An MLP with `activation` between Linear layers and raw logits at the output."""
    act_cls, default_init = ACTIVATIONS[activation]
    init = default_init if init is None else init
    layers = []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(Linear(fan_in, fan_out, rng, init=init, dtype=dtype))
        if i < len(sizes) - 2:
            layers.append(act_cls())
    return Sequential(*layers)
