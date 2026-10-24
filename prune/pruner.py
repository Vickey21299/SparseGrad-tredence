"""Gradual, global, importance-based pruning with optional regrowth."""
import math

import numpy as np

from engine import backend
from prune.criteria import importance
from prune.schedule import cosine_decay, cubic_sparsity


class Pruner:
    """Called after every optimizer step: collects gradient statistics and, on
    the schedule, rewrites the masks of `params` to hit the target sparsity.

    Sparsity is over the prunable parameters only (the weight matrices; biases
    stay dense). Ranking is global across layers. That is fair for the Taylor
    criteria because every score is an estimated loss change in the same units.
    Every layer keeps at least `min_layer_density` of its weights, so no layer
    can be cut off entirely. If the floors add up to more than the budget, the
    floors win and the achieved sparsity (always reported) falls short of the target.

    Regrowth (RigL-style): at each update a fraction of the kept budget, decaying
    from `regrow` to 0 along a cosine, is given to previously pruned connections
    with the largest |dL/dw_eff| (from `Parameter.dense_grad`). They come back at
    exactly 0 with fresh optimizer state (see nn/optim.py), so reviving them does
    not change the network's output at that step.
    """

    def __init__(self, params, final_sparsity, start_step, end_step, every, batch_size,
                 criterion="fisher", regrow=0.0, min_layer_density=0.01):
        self.params = list(params)
        self.final_sparsity = final_sparsity
        self.start_step, self.end_step, self.every = start_step, end_step, every
        self.batch_size = batch_size
        self.criterion = criterion
        self.regrow = regrow
        self.min_layer_density = min_layer_density
        self.total = sum(p.data.size for p in self.params)
        self.history = []
        for p in self.params:
            # An all-True mask from the start, so every step records dense_grad.
            p.set_mask(np.ones(p.shape, dtype=bool))
        self._reset_stats()

    def _reset_stats(self):
        # Kept on the parameters' device in their dtype (float32): two cheap
        # element-wise ops per step. Only mask updates move data to the CPU.
        self.sum_g = [backend.xp_of(p.data).zeros_like(p.data) for p in self.params]
        self.sum_g2 = [backend.xp_of(p.data).zeros_like(p.data) for p in self.params]
        self.count = 0

    def _accumulate(self):
        for p, s, s2 in zip(self.params, self.sum_g, self.sum_g2):
            g = p.dense_grad
            s += g
            s2 += g * g
        self.count += 1

    def is_update_step(self, step):
        if step < self.start_step or step > self.end_step:
            return False
        return (step - self.start_step) % self.every == 0 or step == self.end_step

    def __call__(self, step):
        self._accumulate()
        if self.is_update_step(step):
            self.update(step)

    def update(self, step):
        target = cubic_sparsity(step, self.start_step, self.end_step, self.final_sparsity)
        n_keep = self.total - int(round(target * self.total))
        # Ranking runs on the CPU in float64 (every `every` steps, so the copy is cheap).
        to_cpu = lambda a: backend.to_numpy(a).astype(np.float64)
        g_mean = [to_cpu(s) / self.count for s in self.sum_g]
        g2_mean = [to_cpu(s2) / self.count for s2 in self.sum_g2]
        masks = [backend.to_numpy(p.mask) for p in self.params]

        active = np.concatenate([m.ravel() for m in masks])
        scores, total_floor = [], 0
        for p, m, g, g2 in zip(self.params, masks, g_mean, g2_mean):
            score = importance(self.criterion, to_cpu(p.data), g, g2, self.batch_size).ravel()
            score = np.where(m.ravel(), score, -np.inf)
            # The floor: each layer's best weights are always kept.
            floor = min(math.ceil(self.min_layer_density * p.data.size), int(m.sum()))
            if floor:
                score[np.argpartition(-score, floor - 1)[:floor]] = np.inf
            total_floor += floor
            scores.append(score)
        scores = np.concatenate(scores)
        n_keep = max(n_keep, total_floor)

        n_regrow = int(cosine_decay(step, self.start_step, self.end_step, self.regrow) * n_keep)
        n_regrow = min(n_regrow, int((~active).sum()))
        n_survive = min(n_keep - n_regrow, int(active.sum()))

        new_mask = np.zeros(self.total, dtype=bool)
        if n_survive > 0:
            new_mask[np.argpartition(-scores, n_survive - 1)[:n_survive]] = True
        revived = 0
        if n_keep - n_survive > 0:
            grow_score = np.concatenate([np.abs(g).ravel() for g in g_mean])
            grow_score = np.where(active, -np.inf, grow_score)  # only connections pruned before this update
            grown = np.argpartition(-grow_score, n_keep - n_survive - 1)[:n_keep - n_survive]
            new_mask[grown] = True
            revived = len(grown)

        offset = 0
        for p in self.params:
            p.set_mask(new_mask[offset:offset + p.data.size].reshape(p.shape))
            offset += p.data.size
        self._reset_stats()
        self.history.append({"step": step, "target_sparsity": target, "sparsity": self.sparsity(),
                             "revived": revived})

    def sparsity(self):
        return 1.0 - sum(int(p.mask.sum()) for p in self.params) / self.total

    def layer_densities(self):
        return [round(p.density, 4) for p in self.params]

    def exact_zero_fraction(self):
        """Fraction of prunable weights that are exactly 0.0 in memory (honest-masking check)."""
        return sum(int((p.data == 0.0).sum()) for p in self.params) / self.total
