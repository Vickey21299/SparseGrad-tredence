"""Sparsity and regrowth schedules (functions of the optimizer step)."""
import math


def cubic_sparsity(step, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    """Gradual pruning schedule of Zhu & Gupta (2017).

        s(t) = s_f + (s_i - s_f) * (1 - p)^3,   p = (t - t_start) / (t_end - t_start)

    It prunes fast early, while the network is still redundant and has plenty
    of steps left to recover, and slowly near the end, when every remaining
    connection matters and there is less time to adapt. At half-way it is
    already at 7/8 of the final sparsity.
    """
    if step <= start_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity
    progress = (step - start_step) / (end_step - start_step)
    return final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - progress) ** 3


def cosine_decay(step, start_step, end_step, initial_value):
    """initial_value at start_step, decaying to 0 at end_step (RigL's drop fraction)."""
    if step <= start_step:
        return initial_value
    if step >= end_step:
        return 0.0
    progress = (step - start_step) / (end_step - start_step)
    return 0.5 * initial_value * (1.0 + math.cos(math.pi * progress))
