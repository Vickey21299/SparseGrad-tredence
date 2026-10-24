"""Part 4: the sparsity-accuracy sweep and the importance-vs-magnitude comparison.

    python -m train.sweep --jobs 4              # CPU, 4 runs in parallel (the reference numbers)
    python -m train.sweep --device gpu          # GPU, one run at a time
    python -m train.sweep --analyze-only        # rebuild the tables and plots from finished runs

Every run is `python -m train.prune_mlp` with identical settings (20 epochs, same
schedule) apart from the swept arguments. Finished runs (those with a
summary.json) are skipped, so an interrupted sweep resumes where it stopped.
Writes to results/part4/:
  runs/<name>/      one directory per run (logs, metrics, model)
  sweep_runs.csv    one row per run: the raw numbers
  pareto.csv        mean / std / min / max over seeds, per method and sparsity
  comparison.csv/.md paired per-seed differences with t statistics
  pareto.png        accuracy vs achieved sparsity with error bars
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from train.plots import plot_pareto  # noqa: E402

SPARSITIES = [0.5, 0.75, 0.9, 0.95, 0.98]
REGROW_SPARSITIES = [0.9, 0.95, 0.98]
# Two-sided 95% critical values of Student's t, by degrees of freedom.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def planned_runs(seeds):
    """(method, target sparsity, criterion, regrow, seed) for every run in the sweep.

    The criterion comparison uses regrow 0 for both methods, so the criterion is
    the only difference. Regrowth is an ablation on top of `fisher`. At 0%
    sparsity nothing is pruned, so a single dense run per seed (same 20-epoch
    recipe) is the baseline for every method.
    """
    runs = []
    for seed in seeds:
        runs.append(("dense", 0.0, "magnitude", 0.0, seed))
        for sparsity in SPARSITIES:
            runs.append(("fisher", sparsity, "fisher", 0.0, seed))
            runs.append(("magnitude", sparsity, "magnitude", 0.0, seed))
        for sparsity in REGROW_SPARSITIES:
            runs.append(("fisher+regrow", sparsity, "fisher", 0.2, seed))
    return runs


def run_dir(root, method, sparsity, seed):
    return Path(root) / "runs" / f"{method.replace('+', '_')}_s{round(sparsity * 100):02d}_seed{seed}"


def launch(run, root, device, extra):
    method, sparsity, criterion, regrow, seed = run
    out = run_dir(root, method, sparsity, seed)
    if (out / "summary.json").exists():
        return run, "skipped (done)"
    cmd = [sys.executable, "-m", "train.prune_mlp", "--sparsity", str(sparsity), "--criterion", criterion,
           "--regrow", str(regrow), "--seed", str(seed), "--out", str(out), "--device", device,
           "--no-progress", "--dense-summary", "none", *extra]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return run, "FAILED:\n" + proc.stderr[-2000:]
    return run, f"done in {time.perf_counter() - start:.0f} s"


def collect(root, seeds):
    rows = []
    for method, sparsity, criterion, regrow, seed in planned_runs(seeds):
        path = run_dir(root, method, sparsity, seed) / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        rows.append({
            "method": method, "target_sparsity": sparsity, "seed": seed, "criterion": criterion,
            "regrow": regrow, "achieved_sparsity": s["achieved_sparsity"],
            "exact_zero_fraction": s["exact_zero_fraction"], "active_weights": s["active_weights"],
            "test_acc": s["test_acc"], "test_loss": s["test_loss"],
            "revived_total": s["revived_total"], "train_seconds": s["train_seconds"],
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    """Mean/std over seeds per (method, sparsity); the dense runs are every method's 0% point."""
    out = []
    dense = [r for r in rows if r["method"] == "dense"]
    for method in ("fisher", "magnitude", "fisher+regrow"):
        groups = {0.0: dense} if method != "fisher+regrow" else {}
        for r in rows:
            if r["method"] == method:
                groups.setdefault(r["target_sparsity"], []).append(r)
        for sparsity, group in sorted(groups.items()):
            acc = np.array([g["test_acc"] for g in group])
            out.append({
                "method": method, "target_sparsity": sparsity, "n_seeds": len(group),
                "achieved_sparsity": float(np.mean([g["achieved_sparsity"] for g in group])),
                "active_weights": int(np.mean([g["active_weights"] for g in group])),
                "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                "acc_min": float(acc.min()), "acc_max": float(acc.max()),
            })
    return out


def paired(rows, method_a, method_b):
    """Per-seed differences a - b at each sparsity where both ran with the same seeds."""
    out = []
    by_key = {(r["method"], r["target_sparsity"], r["seed"]): r["test_acc"] for r in rows}
    for sparsity in sorted({r["target_sparsity"] for r in rows if r["method"] == method_a}):
        seeds = sorted(s for (m, sp, s) in by_key if m == method_a and sp == sparsity
                       and (method_b, sparsity, s) in by_key)
        if not seeds:
            continue
        diffs = np.array([by_key[(method_a, sparsity, s)] - by_key[(method_b, sparsity, s)] for s in seeds])
        n = len(diffs)
        sd = diffs.std(ddof=1) if n > 1 else float("nan")
        t = diffs.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
        crit = T_CRIT_95.get(n - 1, 1.96)
        out.append({
            "comparison": f"{method_a} - {method_b}", "target_sparsity": sparsity, "n_seeds": n,
            "mean_diff_pts": 100 * diffs.mean(), "std_diff_pts": 100 * sd,
            "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()),
            "t_stat": t, "t_crit_95": crit,
            "significant_95": bool(np.isfinite(t) and abs(t) > crit),
        })
    return out


def comparison_markdown(comparisons):
    lines = ["| comparison | sparsity | seeds | mean diff (pts) | std (pts) | wins/losses | t | 95% crit | significant |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in comparisons:
        lines.append(f"| {c['comparison']} | {c['target_sparsity']:.0%} | {c['n_seeds']} | "
                     f"{c['mean_diff_pts']:+.3f} | {c['std_diff_pts']:.3f} | {c['wins']}/{c['losses']} | "
                     f"{c['t_stat']:.2f} | {c['t_crit_95']:.2f} | {'yes' if c['significant_95'] else 'no'} |")
    return "\n".join(lines) + "\n"


def analyze(root, seeds):
    root = Path(root)
    rows = collect(root, seeds)
    if not rows:
        print("no finished runs yet")
        return
    write_csv(root / "sweep_runs.csv", rows)
    agg = aggregate(rows)
    write_csv(root / "pareto.csv", agg)
    comparisons = paired(rows, "fisher", "magnitude") + paired(rows, "fisher+regrow", "fisher")
    if comparisons:
        write_csv(root / "comparison.csv", comparisons)
        (root / "comparison.md").write_text(comparison_markdown(comparisons))
    plot_pareto(agg, root / "pareto.png")

    print(f"\n{len(rows)} finished runs -> {root}")
    print(f"{'method':15s} {'target':>7s} {'achieved':>9s} {'active':>7s} {'seeds':>5s}  test acc mean ± std  [min, max]")
    for a in agg:
        print(f"{a['method']:15s} {a['target_sparsity']:7.0%} {a['achieved_sparsity']:9.4f} {a['active_weights']:7d} "
              f"{a['n_seeds']:5d}  {100 * a['acc_mean']:.2f} ± {100 * a['acc_std']:.2f}  "
              f"[{100 * a['acc_min']:.2f}, {100 * a['acc_max']:.2f}]")
    if comparisons:
        print("\n" + comparison_markdown(comparisons))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--jobs", type=int, default=1, help="runs in parallel (CPU; each uses 4 BLAS threads)")
    p.add_argument("--out", default="results/part4")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="further arguments passed to every prune_mlp run (e.g. --extra --epochs 3)")
    args = p.parse_args(argv)

    if not args.analyze_only:
        runs = planned_runs(args.seeds)
        print(f"{len(runs)} runs on {args.device} with {args.jobs} job(s); results in {args.out}/runs")
        jobs = 1 if args.device == "gpu" else args.jobs
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            done = 0
            for run, status in pool.map(lambda r: launch(r, args.out, args.device, args.extra), runs):
                done += 1
                method, sparsity, _, _, seed = run
                print(f"[{done:2d}/{len(runs)}] {method:14s} sparsity {sparsity:.0%} seed {seed}: {status}",
                      flush=True)
    analyze(args.out, args.seeds)


if __name__ == "__main__":
    main()
