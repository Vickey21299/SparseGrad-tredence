# The Self-Pruning Network

A small neural-network library written from scratch in Python + NumPy (reverse-mode
autodiff, layers, Adam), used to train a classifier that prunes its own connections
during training under a hard sparsity budget.

No PyTorch / TensorFlow / JAX / Keras or any autodiff, pruning or NN library is used.
**scikit-learn is used only to load the MNIST dataset** (`fetch_openml`), never for
training, models or gradients.

## Install

With [uv](https://docs.astral.sh/uv/) (preferred):

```bash
uv venv --python 3.10    # any Python >= 3.10
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

## Commands

| What | Command |
|---|---|
| Gradient-check + masked-weight tests (Part 1) | `python -m pytest` |
| Part 2 training (early stopping, ~1 min on CPU) | `python -m train.train_mlp` |
| Evaluate the saved Part 2 model on the test set | `python -m train.test_model` |
| Part 3 self-pruning run (90% sparsity, ~2.5 min) | `python -m train.prune_mlp` |
| Part 4 Pareto sweep (42 runs; ~25 min with 4 parallel CPU jobs) | `python -m train.sweep --jobs 4` |
| Part 4 inference-cost measurement (after the sweep) | `python -m train.cost` |

## Optional: run on an NVIDIA GPU

Every training script accepts `--device gpu`, for example `python -m train.prune_mlp --device gpu`.
It needs CuPy (`uv pip install "cupy-cuda12x[ctk]"`), which provides the NumPy API on CUDA.

- **Same engine on both devices.** Each op takes its array module from its inputs, so
  the engine code is the same on CPU and GPU. Only Adam has a GPU-specific path: the same
  update fused into one kernel, to avoid ~15 kernel launches per parameter.
- **Checked against the CPU.** `tests/test_gpu.py` checks that CPU and GPU training agree
  to ~1e-10, including pruning. It is skipped on machines without CUDA.
- **CPU/NumPy is the reference.** The brief asks for NumPy, and float32 summation order
  differs on the GPU. So all reported numbers come from the default CPU path; GPU runs
  agree to within seed noise.

Measured on an RTX 500 Ada laptop GPU, for one pruning epoch:

| Path | Seconds per epoch |
|---|---|
| CPU, before optimization | 7.2 |
| CPU, optimized | 4.0 |
| GPU | 2.0 |

## Layout

```
engine/   Part 1: Tensor, Parameter (with pruning mask), ops, finite-difference gradcheck
nn/       mask-aware optimizers (SGD+momentum, Adam); layers in Part 2
train/    data loading and training loops (Part 2)
prune/    importance criteria, schedules, mask updates (Part 3)
tests/    gradcheck suite, engine mechanics, masked-weight correctness
```

## Part 1 status

- `Tensor` wraps a NumPy array, records its parents, and stores `.grad`.
- Ops: `+ - * /` (with full broadcasting), `@` (incl. batched), `sum`, `mean`,
  `relu`, `tanh`, `sigmoid`, and fused, numerically stable `softmax_cross_entropy`.
- `backward()` uses an iterative topological sort (no recursion limit) and accumulates
  gradients for re-used nodes; broadcast operands get their gradient summed back to
  their own shape, so `(N, D) + (D,)` gives the bias a column sum.
- Every op is checked against central finite differences (`tests/test_gradcheck.py`).
- Masked weights: see [DESIGN.md](DESIGN.md#the-gradient-of-a-masked-weight) and
  `tests/test_masking.py`.

## Part 2: training a dense MLP

`python -m train.train_mlp` trains a 784-300-100-10 ReLU MLP (LeNet-300-100, 266,610
parameters) on MNIST with the from-scratch Adam (lr 1e-3, batch 128, seed 0). A tqdm bar
shows progress inside each epoch.

**Early stopping:**
- Training runs for at most 1000 epochs.
- It stops once validation loss has not improved by at least `1e-4` for 10 epochs in a
  row (`--patience 10 --min-delta 1e-4`; use `--monitor val_acc` to watch accuracy instead).
- The weights from the best epoch are restored, and only that checkpoint is evaluated on
  the test set. The best epoch is marked on the learning-curve plot.

It writes these files to `results/part2/`:

- `metrics.csv`: per-epoch train/val loss and accuracy, max gradient norm, max |w|
- `learning_curve.png`: the learning-curve plot
- `train.log`, `config.json`, `summary.json`
- `model.npz`: the best checkpoint (not committed)

`python -m train.test_model` reloads that checkpoint and reports:

- test loss and accuracy
- per-class accuracy
- the most common mistake
- `confusion_matrix.png` and `test_report.json`

Early stopping ended the run after 19 epochs. Validation loss was best at epoch 9,
and that checkpoint is the one tested:

| Best checkpoint (epoch 9) | Loss | Accuracy |
|---|---|---|
| Train | 0.019 | 99.34% |
| Validation | 0.084 | 97.68% |
| **Test** | **0.078** | **97.85%** |

![learning curve](results/part2/learning_curve.png)

- **Data:**
  - MNIST, standard 60k/10k split.
  - 5k of the training images are held out for validation, using a fixed split seed.
  - Pixels are scaled to [0, 1], then standardized with training-set mean/std.
  - The test set is evaluated once, at the end.
- **Initialization: He (Kaiming) normal, `std = sqrt(2 / fan_in)`, biases at 0.**
  - **Why:** for `h = relu(W x)` with zero-mean weights, `Var(Wx) = fan_in · Var(w) · E[x²]`. ReLU keeps half of that second moment. So `Var(w) = 2 / fan_in` keeps the activation scale constant from layer to layer.
  - **Measured on 1,000 training images:** the run logs the std of each hidden layer's output at init:

    | Init | Layer 1 | Layer 2 |
    |---|---|---|
    | He | 0.83 | 0.81 (stable) |
    | Xavier | 0.71 | 0.60 (shrinking with ReLU) |
    | N(0, 0.01²) | 0.17 | 0.02 (vanishing) |
    | N(0, 1) | 16.5 | 196 (exploding) |

  - `tests/test_layers.py` checks the same through 6 ReLU layers.
- **Stability:**
  - The loop stops with an error on the first non-finite loss or gradient, and none occurred.
  - The max global gradient norm settles at ~1.5 after epoch 1. Max |w| grows slowly, from 0.47 to 0.52.
- **Honest note:** validation loss is lowest at epoch 4 (0.090) and then creeps up while training loss keeps falling. That is mild overfitting, because there is no regularization. Validation accuracy stays flat at about 97.3–97.6%.
- **Speed:**
  - Training takes ~2.6 s per epoch on CPU.
  - `OPENBLAS_NUM_THREADS` is pinned to 4 in the entry script. With the default of one thread per core, thread overhead made each small matmul ~200× slower: about 26 s per epoch.
- **Optional W&B tracking (all training scripts):**
  - Install it with `uv pip install wandb`, then run `python -m train.train_mlp --wandb offline`. Offline runs are stored under `results/part2/wandb/`; upload one later with `wandb sync <run dir>`.
  - `--wandb online` logs live, after `wandb login`.
  - Every result above is reproduced without W&B.

## Part 3: the network prunes itself

`python -m train.prune_mlp` trains the same MLP from scratch while it removes 90% of the
weight-matrix connections. Biases (410 values) are not pruned and are reported separately.

- **Importance criterion: `½·F·w²`.** This is the second-order Taylor estimate of the
  loss increase from removing a weight, with the Fisher diagonal recovered from
  mini-batch gradient statistics. It was chosen by measuring how well each candidate
  ranks the true loss increase: Spearman 0.94–0.97, versus 0.82–0.93 for magnitude.
  The derivation and the full comparison are in
  [DESIGN.md](DESIGN.md#importance-criterion).
- **Schedule:**
  - 2 epochs of dense warm-up;
  - a cubic sparsity ramp to the target by epoch 12, with mask updates every 50 steps;
  - fine-tuning with a fixed mask until epoch 20.
- **Global ranking across layers,** with a 1% per-layer floor.
- **Regrowth:** 20% of the kept budget, cosine-decayed to 0, goes to pruned connections
  with the largest gradient. They come back at 0 with fresh Adam state.
- **Honest masking:**
  - pruned weights are exact `0.0` in memory, and the run reports the fraction of
    exact zeros next to the mask sparsity;
  - the masked-weight tests from Part 1 guarantee they stay zero.
- **Options:**
  - `--sparsity 0.95`
  - `--criterion magnitude|taylor1|fisher|taylor2`
  - `--regrow 0` to switch regrowth off
  - `--wandb offline`
- **Outputs,** in `results/part3/`:
  - `pruning_run.png`: loss, accuracy and sparsity against the cubic target
  - `metrics.csv`, `mask_updates.json`, `summary.json`, `train.log`

### Results (seed 0, 20 epochs)

| Run | Sparsity (exact zeros) | Active weights | Layer density (L1 / L2 / L3) | Test accuracy |
|---|---|---|---|---|
| Dense, Part 2 (early-stopped, best epoch 9) | 0% | 266,200 | 100 / 100 / 100% | 97.85% |
| `fisher` + regrowth (`results/part3`) | **90.00%** | 26,620 | 7.6 / 27.0 / 74.2% | **98.34%** |
| `fisher` + regrowth (`results/part3_95`) | **95.00%** | 13,310 | 3.4 / 15.2 / 65.1% | **97.99%** |
| `magnitude`, no regrowth (`results/part3_magnitude`) | 90.00% | 26,620 | 7.9 / 25.2 / 55.8% | 98.27% |

- **The hard budget is met exactly,** and the fraction of weights that are exactly `0.0`
  in memory equals the target.
- **The runs reproduce:** rerunning from a clean start with the same code gives identical
  numbers. (These numbers come from the optimized Adam, which rounds very slightly
  differently from the first version, so they were regenerated.)
- Keeping 1 weight in 20 still gives 98.0% on MNIST.
- The global criterion cuts the 784→300 input layer hardest and keeps most of the
  output layer.

**What these numbers do *not* show yet:**

- **That pruning costs nothing, or that it improves accuracy.** The dense baseline is
  an early-stopped checkpoint, while the pruned models trained for 20 epochs. The fair
  accuracy cost is measured against the same 20-epoch recipe at 0% sparsity, which is
  part of the Part 4 sweep.
- **That `fisher` beats `magnitude`.** At 90% they differ by 7 test images out of
  10,000, from one seed, and the two runs also differ in regrowth. Part 4 compares them
  under identical settings over several seeds and at higher sparsity.
- **That regrowth helps.** It swapped 925k connections (mostly early in the ramp, decaying
  to 0 before fine-tuning). Part 4 runs it on versus off.
