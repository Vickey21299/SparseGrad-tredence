# Design

## Importance criterion

**What we want to estimate.** Pruning weight `i` sets it to zero, a perturbation
`δw = −wᵢ eᵢ`. The quantity to estimate is the resulting loss increase
`ΔLᵢ = L(w − wᵢeᵢ) − L(w)`. A second-order Taylor expansion around the current
weights, dropping off-diagonal Hessian terms (Optimal Brain Damage), gives:

```
ΔLᵢ ≈ gᵢ·(−wᵢ) + ½·Hᵢᵢ·wᵢ²          (g = ∂L/∂w,  H = ∇²L)
```

**Getting Hᵢᵢ cheaply.** For a model trained with cross-entropy, the Fisher
information approximates the Hessian (Gauss-Newton argument), and its diagonal is
`F = E_x[(∂ℓₓ/∂wᵢ)²]` over per-example gradients. We only see mini-batch
gradients, which are means of B per-example gradients:

```
E[g_batch²] = ḡ² + σ²/B,   F = ḡ² + σ²   ⇒   F = B·E[g_batch²] − (B−1)·ḡ²
```

`Pruner` collects `ḡ` and `E[g_batch²]` from `dense_grad` over the steps between
two mask updates. That costs two element-wise ops per step, with no extra
backward passes.

**Which terms to keep was decided by measurement.** `tests/test_prune.py` trains
a small MLP, removes every weight of its first layer one at a time, measures the
real `ΔLᵢ`, and compares each criterion's ranking against it. A scratch
experiment extended this to 3 seeds × 2 training lengths and added the exact
Hessian diagonal (from finite differences) as a reference. The numbers are
Spearman rank correlations with the true `ΔL`:

| Model state | `\|w\|` | `−g·w` | `½F·w²` | `½F·w² − g·w` | `½H·w²` | `½H·w² − g·w` |
|---|---|---|---|---|---|---|
| Partly trained (150 steps) | 0.82–0.91 | 0.52–0.59 | **0.95–0.96** | 0.85–0.94 | 0.91–0.98 | 0.92–0.99 |
| Overfit (1500 steps) | 0.88–0.93 | 0.53–0.62 | **0.94–0.97** | 0.61–0.70 | 0.58–0.66 | 0.58–0.66 |

Four conclusions:

1. **The expansion is right when the curvature is right.** With the exact `H`,
   adding the first-order term helps (`H − g·w ≥ H`).
2. **The empirical Fisher underestimates the curvature by 4–5×** (median `F/H`
   ≈ 0.2–0.26). Adding an exact first-order term to an under-scaled second-order
   term gives the first-order term too much weight, which makes `taylor2` worse
   than `F` alone.
3. **The first-order term alone (`|g·w|`) is the weakest criterion.** Near a
   minimum, `E[g] → 0`, so what remains is mini-batch noise.
4. **Close to convergence the local quadratic model breaks down,** even with the
   exact Hessian. Removing a weight is not a small perturbation. `½F·w²` stays
   robust because `F` measures how strongly the model's outputs depend on `wᵢ`,
   which is the quantity that matters.

**Then end-to-end training contradicted it (Part 4).** Across 3 seeds of the full
gradual-pruning pipeline:

- at ≤ 90% sparsity, the criteria tie;
- at 95%, magnitude is ahead by 0.25 points, not significant;
- at 98%, magnitude is ahead by **0.47 points, significant** (t = 9.5, magnitude better
  on 3 of 3 seeds).

Better one-at-a-time damage prediction did not mean better pruning, and the README
discusses why that might be. A single-removal Taylor estimate ignores interactions,
and with fine-tuning what matters is recoverability, not the immediate loss jump. The
lesson I take: validate a criterion on the end-to-end metric, not only on its proxy.

**The default stayed `fisher`: `sᵢ = ½·Fᵢᵢ·wᵢ²`.** This was the choice made before
Part 4. Given the Part 4 result, `--criterion magnitude` is the better production
choice for this model. `taylor2`,
`taylor1` and `magnitude` remain available as ablations and baselines.

**Why it beats magnitude.** Magnitude treats one unit of weight as equally
important everywhere. Fisher weights each `wᵢ²` by how sensitive the loss is to
that connection. A large weight on an input pixel that is always 0 (MNIST
borders) costs nothing to remove, and magnitude still keeps it.

**Global ranking.** `½F·w²` is an estimated loss change, so scores from
different layers are in the same units and can be ranked together. Magnitude
scores cannot be compared that way, because layers have different scales. A
1% per-layer floor stops any layer from being cut off entirely. At 90%,
pruning the 784-300-100-10 MLP leaves the layers at 7% / 31% / 75% density.
The input layer turns out to be the most redundant.

**Schedule.** Pruning uses the cubic ramp of Zhu & Gupta (2017),
`s(t) = s_f·(1 − (1 − p)³)`. Masks are updated every 50 steps, from the end of a
2-epoch dense warm-up until epoch 12. Epochs 12–20 then fine-tune with the mask
fixed. The reasons:

- **Warm-up first:** importance scores from a random network mean nothing.
- **Front-loaded ramp:** pruning is fast early, while the network is redundant
  and has time to recover. It slows near the end, when each connection matters.
- **Fine-tune last:** the network gets to adapt to the final mask.

One-shot pruning after training would need the scores to stay valid while 90%
of the network vanishes at once. A first-order or second-order estimate is only
local, so that assumption breaks.

**Regrowth (RigL-style).** At each mask update, a share of the kept budget is
offered back to pruned connections with the largest `|∂L/∂w_eff|` (from
`dense_grad`). That share starts at 20% and decays along a cosine to 0 by the
end of the ramp. The revived connections:

- start at exactly 0, so the network's output is unchanged at the moment of
  revival;
- get fresh optimizer state and their own bias-correction clock (Part 1);
- must then earn their place under the importance criterion at the next update.

The decay to 0 is there for stability. Without it, swapping connections would
never stop. With it, the mask is frozen for the whole fine-tuning phase.

## The gradient of a masked weight

Every masked parameter `W` has a boolean mask `M`, and the forward pass uses only the
effective weight `W_eff = W ⊙ M` (`Parameter.masked()`). Pruned entries are also
stored as exact zeros in `W`, so a pruned connection is a real zero, not a small number.

**What the engine computes.** By the chain rule,

```
∂L/∂W = ∂L/∂W_eff ⊙ M
```

so `W.grad` is **exactly 0** on pruned entries. That is the true gradient of the
function the network computes: changing a pruned `W_ij` has no effect on the loss.
The tests check it against finite differences for several masks that change between
steps.

The engine also keeps the **dense** gradient `∂L/∂W_eff` in `W.dense_grad`. On a
pruned entry this is non-zero: it is the first-order estimate of how the loss would
change if that connection came back. Only the pruning and regrowth logic uses it. The
optimizer never does, because moving a weight the network does not use would be wrong.

**Why a zero gradient alone is not enough (the subtle bug).** With textbook Adam or
momentum, a zero gradient does not mean a zero update:

- **Stale momentum leaks.** After pruning, `m ← β₁m + 0` stays non-zero for many steps,
  so the update `m̂/√v̂` keeps moving the "pruned" weight away from 0. This is the
  `test_textbook_adam_leaks_pruned_weights_even_with_zero_gradient` test.
- **Stale state on revival.** If the moments are kept frozen, a revived connection
  starts with momentum from a different point in training: other weights have moved,
  so the old direction no longer applies.
- **Global-step bias correction.** Even if the moments are reset by hand, Adam's bias
  correction uses the global step `t`. For large `t` that correction is about 1, so the
  revived entry's first step is `(1−β₁)/√(1−β₂) ≈ 3.16×` larger than a fresh Adam
  step (`test_textbook_adam_overshoots_first_step_after_revival`).

**What the optimizers do** (`nn/optim.py`):

1. Ignore the gradient on pruned entries, even if a caller bypassed `masked()`.
2. Hold momentum, the Adam moments and a **per-entry step count** at 0 while an entry
   is pruned. This includes entries that were pruned and revived between two optimizer
   steps (tracked by `Parameter.pruned_since_step`).
3. Apply Adam's bias correction using each entry's own step count.

As a result, a pruned weight stays exactly 0 for every step. A revived connection
follows the same trajectory as a brand-new connection that starts at 0. Connections
that were never pruned follow exactly the trajectory of an unpruned run.
`test_revived_connection_behaves_like_a_brand_new_connection` checks all three. A
mutation run showed that removing any of the resets, or using the global step count,
makes the suite fail.

Reviving at 0 with empty state (as in RigL) is a deliberate choice. The connection
re-enters the network without changing the function at that step, so it cannot cause a
jump in the loss. It then has to earn a weight from its own gradients.

## Engine bottlenecks and optimizations

These were measured on the 784-300-100-10 MLP with batch 128 (430 steps per epoch).
Each bottleneck was found by profiling, not by guessing.

| Bottleneck | Evidence | Fix | Effect |
|---|---|---|---|
| BLAS threading | One thread per core (22) turned a 128×784·784×300 matmul from 0.3 ms into 50 ms of thread overhead | Pin `OPENBLAS_NUM_THREADS=4` | Dense epoch 26 s → 2.6 s |
| Adam's per-entry bias correction | `β ** t` evaluated per element on every step for masked parameters | Keep `βᵗ` per entry and update it with one multiply | Pruning epoch 7.2 s → 4.0 s, together with the two fixes below |
| The mask re-converted to float on every forward | `mask.astype(float)` on every `masked()` call | Cache the float mask in `set_mask` | (included above) |
| Pruner statistics in float64 | A float64 copy of each gradient on every step | float32 sums on the device; float64 only at mask updates | (included above) |
| GPU host↔device syncs | About 9 `float()`/`int()` syncs per step | Fetch the loss, gradient norm and #correct in a single copy | GPU epoch 29 s* → 4.2 s → 2.0 s, together with the fix below |
| GPU kernel launches | Element-wise Adam = ~15 tiny kernels per parameter | One fused `ElementwiseKernel` per parameter | (included above) |

\*The first GPU epoch also includes one-off CUDA kernel compilation, which is cached
afterwards.

**What still bottlenecks it:**

- **Python overhead per op.** Every op builds a new graph node and closure, and every
  result is a newly allocated array. For a model this small, that, not FLOPs, is the
  limit. It is why the GPU, despite matmuls 16× faster than the CPU, is only 2× faster
  end to end.
- **Remedies:** the next steps would be
  - fused layers (linear + bias + ReLU as one op with a hand-written backward);
  - reusing gradient buffers in place instead of allocating new ones;
  - larger batches on the GPU;
  - recording the graph once and replaying it, since the structure is identical every
    step and only the mask values change.
- **Dense compute on pruned weights.** The masked forward pass is still a dense
  `x @ (W ⊙ M)`, so at 90% sparsity 90% of the multiply-adds are multiplications by
  zero. A real saving needs a sparse-aware forward pass. That is measured in Part 4.

## Serving a self-pruned model at scale

**First, what Part 4 measured.** The sparsity only becomes speed if the served model's
format exploits it:

- **Unstructured CSR in NumPy:** slower than dense at every sparsity.
- **Structural compaction** (dropping dead inputs and units): 2.7× faster at 98%.

So the serving plan starts by turning unstructured sparsity into a hardware-friendly
form.

**1. Export: a compile step, not the training graph.** After training, the model goes
through an offline "compile":

- Apply the masks.
- Run `compact_mlp`: drop dead inputs and units, and fold constant units into the
  biases. The result is smaller dense matrices.
- Choose the format per layer, **by benchmark on the target hardware, not by
  sparsity alone**:
  - dense BLAS if density > ~10%;
  - a compiled CSR/BSR kernel (MKL, cuSPARSE) below that;
  - on GPUs, N:M (2:4) structured sparsity if the pruning was constrained to it, since
    only that pattern runs on Ampere+ sparse tensor cores.
- Optionally quantize to int8. That gives roughly another 4× on memory and bandwidth,
  and pruning plus quantization compose.

The artifact (weights, format, input index map, a hash of the code version) goes into
a versioned model registry, alongside the validation report: accuracy, per-class
accuracy, latency, and an equivalence check against the training-time masked model.
`train/cost.py` is a small version of that check: it asserts identical predictions
before it times anything.

**2. The runtime: throughput through batching, latency through limits.**

- **Batching:** a request router places requests on per-model queues, and a dynamic
  batcher accumulates up to B requests or T milliseconds. Our measurements show
  per-sample cost falling ~5× from batch 1 to batch 64, because the fixed ~15–20 µs
  per call is amortised. Batching is worth more than any kernel trick at this model size.
- **Threads:** pin a fixed number of BLAS threads per worker. Part 2 showed that the
  default of one thread per core made each small matmul ~200× slower. In a
  multi-tenant process, oversubscription is the default failure mode. Run one worker
  per core group, with a fixed thread count.
- **Scale and scheduling:** at thousands of requests per second, a 1-ms batch window
  holds dozens of requests, so the batcher stays full. Autoscaling uses queue depth and
  p99 latency, not CPU utilisation.

**3. Multi-tenancy: what pruning actually buys.** A 98%-pruned model in CSR is 45 KB
instead of 1 MB. The win is **density, not raw speed**:

- far more tenants' models fit in RAM or GPU memory per node;
- an LRU cache of hot models avoids cold loads;
- per-tenant quotas and isolated queues stop one noisy tenant from consuming the
  shared batcher;
- models that share an architecture can be co-batched; with structurally compacted
  models whose shapes differ, batch per model.

**4. Safety in rollout.**

- **Canary and shadow traffic:** compare the pruned model against the dense one on live
  traffic, using:
  - agreement rate;
  - per-class accuracy, where pruning can hurt rare classes while the average looks fine;
  - calibration.
- **Monitoring:** watch for input-distribution drift. Pruned capacity was removed for
  the *training* distribution, so a pruned model may degrade earlier under shift.
- **Rollback:** a registry pointer flip back to the previous model version.

**What I would not do:** serve the training-time `x @ (W ⊙ M)` graph. It pays the full
dense cost while storing the mask in addition, so it is strictly worse than the dense
model.
