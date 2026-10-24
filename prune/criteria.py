"""Importance criteria: how much would the loss go up if a weight were removed?

Removing weight i means the perturbation dw_i = -w_i. A second-order Taylor
expansion of the loss around the current weights gives

    dL_i ~= g_i * (-w_i) + 1/2 * H_ii * w_i^2

(off-diagonal Hessian terms dropped, as in Optimal Brain Damage). The full
derivation is in DESIGN.md. Every criterion returns a score where higher means
"more important, keep it".

Default: "fisher". Against the true loss increase measured by actually removing
each weight (tests/test_prune.py, DESIGN.md) it ranked best: Spearman 0.94-0.97
over 6 runs, versus 0.82-0.93 for magnitude. "taylor2" adds the first-order
term. That is exact in theory, but the empirical Fisher underestimates the
true curvature ~4-5x, so the sum gives the first-order term too much weight.
"""
import numpy as np

CRITERIA = ("magnitude", "taylor1", "fisher", "taylor2")


def fisher_diagonal(g_mean, g2_mean, batch_size):
    """Per-example empirical Fisher diagonal, recovered from mini-batch gradient statistics.

    A mini-batch gradient is the mean of B per-example gradients, so
    E[g_batch^2] = mean^2 + var/B. The per-example Fisher F = E[g_x^2] =
    mean^2 + var, so F = B * E[g_batch^2] - (B - 1) * mean^2. F approximates
    the Hessian diagonal H_ii for a model trained with cross-entropy.
    """
    return np.maximum(batch_size * g2_mean - (batch_size - 1) * g_mean ** 2, 0.0)


def importance(criterion, w, g_mean, g2_mean, batch_size):
    """Score each entry of `w` from gradient statistics averaged over recent steps.

    magnitude: |w|. The baseline: assumes every weight matters equally per unit size.
    taylor1:   |g w|. First-order term only. Near a minimum E[g] -> 0, so it
               becomes mostly noise.
    fisher:    1/2 F w^2. The second-order term only (OBD with the Fisher as the Hessian).
    taylor2:   1/2 F w^2 - g w. Both terms: the estimated loss increase itself.
    """
    if criterion == "magnitude":
        return np.abs(w)
    if criterion == "taylor1":
        return np.abs(g_mean * w)
    second_order = 0.5 * fisher_diagonal(g_mean, g2_mean, batch_size) * w * w
    if criterion == "fisher":
        return second_order
    if criterion == "taylor2":
        return second_order - g_mean * w
    raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")
