"""Evaluate a trained model checkpoint on the MNIST test set.

    python -m train.test_model                         # uses results/part2
    python -m train.test_model --run results/try1 --split val

Rebuilds the architecture from <run>/config.json, loads <run>/model.npz, and
reports loss, accuracy, per-class accuracy and the confusion matrix. Writes
<run>/test_report.json and <run>/confusion_matrix.png.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # see train_mlp.py

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from engine import Tensor, backend, no_grad  # noqa: E402
from nn import mlp  # noqa: E402
from train.data import load_mnist, to_device  # noqa: E402
from train.loop import evaluate  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="results/part2", help="run directory with config.json and model.npz")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--data-home", default="data")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    return p.parse_args(argv)


def load_model(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    model = mlp(config["sizes"], np.random.default_rng(0), activation=config["activation"])
    with np.load(run_dir / "model.npz") as state:
        model.load_state_dict(state)
    return model, config


def predict(model, X, batch_size=2000):
    with no_grad():
        return np.concatenate([backend.to_numpy(model(Tensor(X[i:i + batch_size])).data.argmax(axis=1))
                               for i in range(0, len(X), batch_size)])


def plot_confusion(confusion, path, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(confusion, cmap="Blues")
    for (i, j), count in np.ndenumerate(confusion):
        ax.text(j, i, count, ha="center", va="center", fontsize=7,
                color="white" if count > confusion.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run)
    backend.set_device(args.device)
    model, config = load_model(run_dir)
    X, y = to_device(load_mnist(args.data_home))[args.split]

    loss, acc = evaluate(model, X, y)
    pred = predict(model, X)
    y = backend.to_numpy(y)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    per_class = confusion.diagonal() / confusion.sum(axis=1)

    print(f"model {config['sizes']} from {run_dir}")
    print(f"{args.split}: {len(y)} images  loss {loss:.4f}  accuracy {acc:.4f}  "
          f"({len(y) - int(confusion.trace())} wrong)")
    print("per-class accuracy:")
    for digit, a in enumerate(per_class):
        print(f"  {digit}: {a:.4f}  ({confusion[digit].sum()} images)")
    off_diagonal = confusion - np.diag(confusion.diagonal())
    true, guess = np.unravel_index(off_diagonal.argmax(), off_diagonal.shape)
    print(f"most common mistake: {true} predicted as {guess} ({off_diagonal[true, guess]} times)")

    plot_confusion(confusion, run_dir / "confusion_matrix.png", f"{args.split} accuracy {acc:.2%}")
    report = {
        "split": args.split,
        "loss": loss,
        "accuracy": acc,
        "per_class_accuracy": {str(d): float(a) for d, a in enumerate(per_class)},
        "confusion_matrix": confusion.tolist(),
    }
    (run_dir / f"{args.split}_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {run_dir / 'confusion_matrix.png'} and {run_dir / f'{args.split}_report.json'}")


if __name__ == "__main__":
    main()
