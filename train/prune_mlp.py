"""Part 3: train the MLP so that it prunes itself down to a target sparsity.

    python -m train.prune_mlp                                # 90% sparsity, fisher + regrowth
    python -m train.prune_mlp --sparsity 0.95 --criterion magnitude --regrow 0
    python -m train.prune_mlp --wandb offline

Timeline (epochs): dense warm-up -> cubic pruning ramp -> fine-tune with a fixed mask.
Early stopping is not used: the schedule is tied to the step count, and only a
model at the final sparsity may be reported. The final model is tested once.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # see train_mlp.py

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from engine import backend  # noqa: E402
from nn import Adam, Linear, mlp  # noqa: E402
from prune import CRITERIA, Pruner, cubic_sparsity  # noqa: E402
from train.data import load_mnist, to_device  # noqa: E402
from train.logger import RunLogger  # noqa: E402
from train.loop import evaluate, train  # noqa: E402
from train.plots import plot_pruning_run  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sparsity", type=float, default=0.9, help="fraction of weight-matrix entries to remove")
    p.add_argument("--criterion", choices=CRITERIA, default="fisher")
    p.add_argument("--regrow", type=float, default=0.2,
                   help="initial fraction of the kept budget offered to regrowth (cosine-decayed to 0); 0 disables")
    p.add_argument("--min-layer-density", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=float, default=2, help="dense training before pruning starts")
    p.add_argument("--prune-end-epoch", type=float, default=12, help="target sparsity reached here")
    p.add_argument("--prune-every", type=int, default=50, help="steps between mask updates")
    p.add_argument("--hidden", type=int, nargs="+", default=[300, 100])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-home", default="data")
    p.add_argument("--out", default="results/part3")
    p.add_argument("--dense-summary", default="results/part2/summary.json",
                   help="Part 2 summary to compare against, if it exists")
    p.add_argument("--wandb", choices=["off", "offline", "online"], default="off")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="gpu runs the same engine on CuPy; cpu (NumPy) is the reference")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = vars(args).copy()
    rng = np.random.default_rng(args.seed)
    backend.set_device(args.device)

    data = to_device(load_mnist(args.data_home))
    sizes = [data["train"][0].shape[1], *args.hidden, 10]
    steps_per_epoch = math.ceil(len(data["train"][0]) / args.batch_size)
    start_step = int(args.warmup_epochs * steps_per_epoch)
    end_step = int(args.prune_end_epoch * steps_per_epoch)
    config.update(sizes=sizes, steps_per_epoch=steps_per_epoch, start_step=start_step, end_step=end_step)
    logger = RunLogger(args.out, config, wandb_mode=None if args.wandb == "off" else args.wandb,
                       name=f"part3-{args.criterion}-s{args.sparsity}-regrow{args.regrow}-seed{args.seed}")

    model = mlp(sizes, rng)
    weights = [m.weight for m in model.modules() if isinstance(m, Linear)]
    pruner = Pruner(weights, args.sparsity, start_step, end_step, args.prune_every, args.batch_size,
                    criterion=args.criterion, regrow=args.regrow, min_layer_density=args.min_layer_density)
    optimizer = Adam(model.parameters(), lr=args.lr)
    logger.info(f"device: {backend.device_name()}")
    logger.info(f"model {sizes}: {pruner.total:,} prunable weights; prune from step {start_step} "
                f"to {end_step} every {args.prune_every} steps to {args.sparsity:.0%} with {args.criterion}, "
                f"regrow {args.regrow}")

    def epoch_metrics():
        return {"sparsity": pruner.sparsity(), "revived": sum(h["revived"] for h in pruner.history)}

    start = time.perf_counter()
    result = train(model, optimizer, data, args.epochs, args.batch_size, rng, logger,
                   after_step=pruner, extra_epoch_metrics=epoch_metrics, progress=not args.no_progress)
    train_seconds = time.perf_counter() - start

    test_loss, test_acc = evaluate(model, *data["test"])
    active = sum(int(w.mask.sum()) for w in weights)
    n_bias = sum(p.data.size for p in model.parameters()) - pruner.total
    logger.info(f"test: loss {test_loss:.4f}  acc {test_acc:.4f}  sparsity {pruner.sparsity():.4f}  "
                f"exact zeros {pruner.exact_zero_fraction():.4f}  per-layer density {pruner.layer_densities()}")

    dense_acc = None
    dense_path = Path(args.dense_summary)
    if dense_path.exists():
        dense_acc = json.loads(dense_path.read_text())["test_acc"]

    sched_steps = np.arange(0, args.epochs * steps_per_epoch + 1, steps_per_epoch // 10)
    schedule = (sched_steps, [cubic_sparsity(s, start_step, end_step, args.sparsity) for s in sched_steps])
    plot_path = logger.run_dir / "pruning_run.png"
    plot_pruning_run(result, pruner.history, schedule, plot_path, dense_acc=dense_acc,
                     title=f"Self-pruning to {args.sparsity:.0%} ({args.criterion}, regrow {args.regrow}, "
                           f"seed {args.seed})")
    logger.log_image("pruning_run", plot_path)
    (logger.run_dir / "mask_updates.json").write_text(json.dumps(pruner.history, indent=1))
    np.savez(logger.run_dir / "model.npz", **model.state_dict(),
             **{f"mask{i}": backend.to_numpy(w.mask) for i, w in enumerate(weights)})

    summary = {
        "criterion": args.criterion,
        "regrow": args.regrow,
        "target_sparsity": args.sparsity,
        "achieved_sparsity": pruner.sparsity(),
        "exact_zero_fraction": pruner.exact_zero_fraction(),
        "layer_density": pruner.layer_densities(),
        "active_weights": active,
        "prunable_weights": pruner.total,
        "dense_biases": n_bias,
        "revived_total": sum(h["revived"] for h in pruner.history),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "dense_test_acc": dense_acc,
        "accuracy_cost": None if dense_acc is None else dense_acc - test_acc,
        "train_seconds": round(train_seconds, 1),
    }
    logger.finish(summary)
    return summary


if __name__ == "__main__":
    main()
