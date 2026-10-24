"""Part 2: train a dense MLP on MNIST with the from-scratch engine and Adam.

    python -m train.train_mlp                      # reproduces results/part2
    python -m train.train_mlp --wandb offline      # also record a W&B run
"""
import os

# Must be set before NumPy loads OpenBLAS. With one thread per core (22 here),
# a 128x784 @ 784x300 matmul took ~50 ms from thread overhead; 4 threads take
# ~0.3 ms. A fixed count also keeps float summation order reproducible.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from engine import Tensor, backend, no_grad
from nn import SGD, Adam, Linear, mlp
from train.data import load_mnist, to_device
from train.logger import RunLogger
from train.loop import EarlyStopping, evaluate, train
from train.plots import plot_learning_curve


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hidden", type=int, nargs="+", default=[300, 100])
    p.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--epochs", type=int, default=1000, help="upper limit; early stopping usually ends sooner")
    p.add_argument("--monitor", choices=["val_loss", "val_acc"], default="val_loss",
                   help="metric watched by early stopping and used to pick the best checkpoint")
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without improvement")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="smallest change in the monitored metric that counts as an improvement")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-home", default="data")
    p.add_argument("--out", default="results/part2")
    p.add_argument("--wandb", choices=["off", "offline", "online"], default="off")
    p.add_argument("--no-progress", action="store_true", help="hide the tqdm progress bar")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="gpu runs the same engine on CuPy; cpu (NumPy) is the reference")
    return p.parse_args(argv)


def activation_scales(X, sizes, activation, seed):
    """Std of each hidden layer's output at initialization, for several init schemes.

    Evidence for the init choice: with He init the scale stays ~constant with
    depth, while too-small / too-large fixed scales shrink or blow up.
    """
    report = {}
    for scheme in ("he", "xavier", 0.01, 1.0):
        model = mlp(sizes, np.random.default_rng(seed), activation=activation, init=scheme)
        h, stds = Tensor(X), []
        with no_grad():
            for layer in model.layers:
                h = layer(h)
                if not isinstance(layer, Linear):
                    stds.append(round(float(h.data.std()), 4))
        report[str(scheme)] = stds
    return report


def main(argv=None):
    args = parse_args(argv)
    config = vars(args).copy()
    rng = np.random.default_rng(args.seed)

    backend.set_device(args.device)
    data = to_device(load_mnist(args.data_home))
    sizes = [data["train"][0].shape[1], *args.hidden, 10]
    config["sizes"] = sizes
    logger = RunLogger(args.out, config, wandb_mode=None if args.wandb == "off" else args.wandb,
                       name=f"part2-{args.activation}-{args.optimizer}-seed{args.seed}")
    logger.info(f"device: {backend.device_name()}")
    logger.info(f"train {data['train'][0].shape}  val {data['val'][0].shape}  test {data['test'][0].shape}")

    scales = activation_scales(data["train"][0][:1000], sizes, args.activation, args.seed)
    logger.info("hidden activation std at init, per layer: " + json.dumps(scales))

    model = mlp(sizes, rng, activation=args.activation)
    n_params = sum(p.data.size for p in model.parameters())
    logger.info(f"model {sizes}, {n_params:,} parameters")
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Early stopping also keeps the best epoch's weights, so the model that is
    # tested is the best one on validation, not the last (possibly overfit) one.
    stopper = EarlyStopping(model, logger, monitor=args.monitor, patience=args.patience,
                            min_delta=args.min_delta)

    start = time.perf_counter()
    result = train(model, optimizer, data, args.epochs, args.batch_size, rng, logger,
                   after_epoch=stopper, progress=not args.no_progress)
    train_seconds = time.perf_counter() - start

    final = result["epochs"][-1]
    best_record = result["epochs"][stopper.best_epoch - 1]
    np.savez(logger.run_dir / "model.npz", **stopper.restore())
    test_loss, test_acc = evaluate(model, *data["test"])
    logger.info(f"test (best checkpoint, epoch {stopper.best_epoch}): loss {test_loss:.4f}  acc {test_acc:.4f}")

    plot_path = logger.run_dir / "learning_curve.png"
    plot_learning_curve(result, plot_path, best_epoch=stopper.best_epoch,
                        title=f"MLP {sizes} on MNIST ({args.optimizer}, seed {args.seed})")
    logger.log_image("learning_curve", plot_path)

    logger.finish({
        "parameters": n_params,
        "max_epochs": args.epochs,
        "epochs_run": final["epoch"],
        "stopped_early": final["epoch"] < args.epochs,
        "monitor": args.monitor,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": best_record["val_loss"],
        "best_val_acc": best_record["val_acc"],
        "final_val_acc": final["val_acc"],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_seconds": round(train_seconds, 1),
        "init_activation_std": scales,
    })
    return result


if __name__ == "__main__":
    main()
