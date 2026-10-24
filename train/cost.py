"""Part 4: does pruning make inference cheaper? Measured, not assumed.

    python -m train.cost                       # uses the finished sweep runs (seed 0, fisher)

For each sparsity it loads the trained, pruned model and runs the MNIST test
set through three CPU forward paths, after checking they give identical
predictions:

  dense    x @ (W*M): the dense-times-zero shortcut used in training. The
           pruned zeros are still multiplied, so this is NOT a real speed-up.
  csr      prune/sparse.py CSRLinear: one multiply-add per surviving weight.
  compact  prune/sparse.py compact_mlp: dead inputs and units removed,
           leaving smaller dense matrices.

Reports multiply-accumulates (MACs) per sample, stored bytes and wall-clock
latency per sample at batch sizes 1, 64 and 1024 (median of 5 repeats).
Writes results/part4/cost.csv, cost.md and cost.png.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # same as training; see train_mlp.py

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from prune.sparse import CSRMLP, compact_mlp, dense_forward  # noqa: E402
from train.data import load_mnist  # noqa: E402
from train.plots import plot_cost  # noqa: E402
from train.sweep import run_dir  # noqa: E402

BATCHES = (1, 64, 1024)


def load_layers(path):
    """[(W * M, b), ...] as float32 from a prune_mlp model.npz."""
    with np.load(path) as f:
        n_layers = sum(1 for k in f.files if k.startswith("mask"))
        return [((f[f"p{2 * i}"] * f[f"mask{i}"]).astype(np.float32), f[f"p{2 * i + 1}"].astype(np.float32))
                for i in range(n_layers)]


def time_per_sample(fn, X, batch, min_seconds=0.2, repeats=5):
    """Median over `repeats` of the mean time per sample, each repeat lasting >= min_seconds."""
    xb = np.ascontiguousarray(X[:batch])
    fn(xb)  # warm-up
    results = []
    for _ in range(repeats):
        calls, start = 0, time.perf_counter()
        while True:
            fn(xb)
            calls += 1
            elapsed = time.perf_counter() - start
            if elapsed >= min_seconds:
                break
        results.append(elapsed / (calls * batch))
    return float(np.median(results)) * 1e6


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="results/part4")
    p.add_argument("--method", default="fisher")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sparsities", type=float, nargs="+", default=[0.0, 0.5, 0.75, 0.9, 0.95, 0.98])
    p.add_argument("--data-home", default="data")
    args = p.parse_args(argv)

    X, y = load_mnist(args.data_home)["test"]
    rows, table = [], []
    for sparsity in args.sparsities:
        method = "dense" if sparsity == 0 else args.method
        path = run_dir(args.sweep, method, sparsity, args.seed) / "model.npz"
        if not path.exists():
            print(f"skip {sparsity:.0%}: {path} not found (run the sweep first)")
            continue
        layers = load_layers(path)
        csr, compact = CSRMLP(layers), compact_mlp(layers)
        paths = {"dense": lambda x, L=layers: dense_forward(L, x.copy()), "csr": csr, "compact": compact}

        # Same function: identical predictions on the full test set.
        reference = paths["dense"](X)
        preds = {name: fn(X).argmax(axis=1) for name, fn in paths.items()}
        for name, fn in paths.items():
            max_diff = float(np.abs(fn(X) - reference).max())
            assert max_diff < 1e-3, (name, max_diff)
            assert (preds[name] == preds["dense"]).all(), name
        accuracy = float((preds["dense"] == y).mean())

        nnz = sum(int(np.count_nonzero(W)) for W, _ in layers)
        dense_macs = sum(W.size for W, _ in layers)
        compact_macs = sum(a * b for a, b in compact.shapes)
        macs = {"dense": dense_macs, "csr": nnz, "compact": compact_macs}
        nbytes = {"dense": sum(W.nbytes + b.nbytes for W, b in layers), "csr": csr.nbytes(), "compact": compact.nbytes()}
        for batch in BATCHES:
            for name, fn in paths.items():
                rows.append({"sparsity": sparsity, "method": name, "batch": batch,
                             "us_per_sample": time_per_sample(fn, X, batch),
                             "macs_per_sample": macs[name], "bytes": nbytes[name], "test_acc": accuracy})
        shape = " -> ".join([str(len(compact.input_index))] + [str(s[1]) for s in compact.shapes])
        table.append({"sparsity": sparsity, "nnz": nnz, "compact_shape": shape, "acc": accuracy})
        latest = {r["method"] + str(r["batch"]): r["us_per_sample"] for r in rows if r["sparsity"] == sparsity}
        print(f"{sparsity:4.0%}  nnz {nnz:6d}  compact {shape:22s}  acc {accuracy:.4f}  "
              + "  ".join(f"b{b}: dense {latest['dense' + str(b)]:.2f} csr {latest['csr' + str(b)]:.2f} "
                          f"compact {latest['compact' + str(b)]:.2f} µs" for b in BATCHES))

    if not rows:
        return
    out = Path(args.sweep)
    with open(out / "cost.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_cost(rows, out / "cost.png")

    dense_us = {(r["batch"]): r["us_per_sample"] for r in rows if r["sparsity"] == rows[0]["sparsity"] and r["method"] == "dense"}
    lines = ["| sparsity | surviving weights (MACs/sample) | compact shape | KB dense / CSR / compact | "
             + " | ".join(f"µs/sample @ batch {b}: dense / csr / compact" for b in BATCHES) + " |",
             "|---|---|---|---|" + "---|" * len(BATCHES)]
    for t in table:
        sel = {(r["method"], r["batch"]): r for r in rows if r["sparsity"] == t["sparsity"]}
        kb = " / ".join(f"{sel[(m, BATCHES[0])]['bytes'] / 1024:.0f}" for m in ("dense", "csr", "compact"))
        cells = [" / ".join(f"{sel[(m, b)]['us_per_sample']:.2f}" for m in ("dense", "csr", "compact")) for b in BATCHES]
        lines.append(f"| {t['sparsity']:.0%} | {t['nnz']:,} | {t['compact_shape']} | {kb} | " + " | ".join(cells) + " |")
    (out / "cost.md").write_text("\n".join(lines) + "\n")
    (out / "cost_meta.json").write_text(json.dumps({
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "batches": BATCHES,
        "dense_baseline_us_per_sample": dense_us, "model": f"{args.method} seed {args.seed}"}, indent=2))
    print(f"\nwrote {out / 'cost.csv'}, {out / 'cost.md'}, {out / 'cost.png'}")


if __name__ == "__main__":
    main()
