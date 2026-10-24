"""Plotting helpers (matplotlib, written to files, no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_learning_curve(result, path, title="Learning curve", best_epoch=None):
    epochs = result["epochs"]
    steps, batch_losses = np.array(result["steps"]).T
    steps_per_epoch = epochs[0]["step"]
    ep = [r["epoch"] for r in epochs]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(steps / steps_per_epoch, batch_losses, color="0.75", lw=0.6, label="mini-batch loss")
    ax_loss.plot(ep, [r["train_loss"] for r in epochs], "o-", label="train (epoch mean)")
    ax_loss.plot(ep, [r["val_loss"] for r in epochs], "s-", label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", yscale="log", title="Loss")
    ax_loss.legend()

    ax_acc.plot(ep, [100 * r["train_acc"] for r in epochs], "o-", label="train")
    ax_acc.plot(ep, [100 * r["val_acc"] for r in epochs], "s-", label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    ax_acc.legend()
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, color="green", ls="--", lw=1, label=f"best epoch ({best_epoch})")
            ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


PARETO_STYLE = {
    "fisher": dict(marker="o", color="tab:blue", label="fisher  ½F·w² (no regrowth)"),
    "magnitude": dict(marker="s", color="tab:orange", label="magnitude |w| (no regrowth)"),
    "fisher+regrow": dict(marker="^", color="tab:green", label="fisher + regrowth"),
}


def plot_pareto(aggregated, path):
    """Test accuracy vs achieved sparsity, mean ± std over seeds; right panel zooms on the high-sparsity end."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, zoom in zip(axes, (False, True)):
        for method, style in PARETO_STYLE.items():
            pts = [a for a in aggregated if a["method"] == method and (not zoom or a["target_sparsity"] >= 0.9)]
            if not pts:
                continue
            ax.errorbar([100 * a["achieved_sparsity"] for a in pts], [100 * a["acc_mean"] for a in pts],
                        yerr=[100 * a["acc_std"] for a in pts], capsize=3, lw=1.2, ms=5, **style)
        dense = [a for a in aggregated if a["method"] == "fisher" and a["target_sparsity"] == 0.0]
        if dense:
            d = dense[0]
            ax.axhline(100 * d["acc_mean"], color="k", ls=":", lw=1,
                       label=f"dense, same recipe ({100 * d['acc_mean']:.2f} ± {100 * d['acc_std']:.2f}%)")
            ax.axhspan(100 * (d["acc_mean"] - d["acc_std"]), 100 * (d["acc_mean"] + d["acc_std"]),
                       color="k", alpha=0.06)
        ax.set(xlabel="achieved sparsity of weight matrices (%)", ylabel="test accuracy (%)",
               title="high-sparsity end (≥ 90%)" if zoom else "sparsity–accuracy trade-off")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    n = max((a["n_seeds"] for a in aggregated), default=0)
    fig.suptitle(f"MNIST MLP 784-300-100-10: mean ± std over {n} seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cost(rows, path):
    """Inference latency per sample vs sparsity, for each forward path and batch size."""
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4), squeeze=False)
    for ax, batch in zip(axes[0], batches):
        for method, style in (("dense", "o-"), ("csr", "s-"), ("compact", "^-")):
            pts = sorted((r["sparsity"], r["us_per_sample"]) for r in rows if r["batch"] == batch and r["method"] == method)
            ax.plot([100 * s for s, _ in pts], [t for _, t in pts], style, label=method)
        ax.set(xlabel="sparsity (%)", ylabel="µs per sample", yscale="log", title=f"batch {batch}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Measured CPU inference cost (median of repeats)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pruning_run(result, prune_history, schedule, path, title="Self-pruning run", dense_acc=None):
    """Loss, accuracy and sparsity against epoch for a pruning run.

    schedule: (steps, target_sparsity) arrays for the planned cubic ramp.
    """
    epochs = result["epochs"]
    steps_per_epoch = epochs[0]["step"]
    ep = [r["epoch"] for r in epochs]

    fig, (ax_loss, ax_acc, ax_sp) = plt.subplots(1, 3, figsize=(15, 4))
    steps, batch_losses = np.array(result["steps"]).T
    ax_loss.plot(steps / steps_per_epoch, batch_losses, color="0.75", lw=0.6, label="mini-batch loss")
    ax_loss.plot(ep, [r["train_loss"] for r in epochs], "o-", ms=3, label="train (epoch mean)")
    ax_loss.plot(ep, [r["val_loss"] for r in epochs], "s-", ms=3, label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", yscale="log", title="Loss")

    ax_acc.plot(ep, [100 * r["train_acc"] for r in epochs], "o-", ms=3, label="train")
    ax_acc.plot(ep, [100 * r["val_acc"] for r in epochs], "s-", ms=3, label="validation")
    if dense_acc is not None:
        ax_acc.axhline(100 * dense_acc, color="k", ls=":", lw=1, label=f"dense test ({dense_acc:.2%})")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")

    sched_steps, sched_target = schedule
    ax_sp.plot(np.asarray(sched_steps) / steps_per_epoch, 100 * np.asarray(sched_target), "--",
               color="0.4", label="cubic target")
    ax_sp.plot([h["step"] / steps_per_epoch for h in prune_history],
               [100 * h["sparsity"] for h in prune_history], ".", ms=4, label="achieved (mask updates)")
    ax_sp.set(xlabel="epoch", ylabel="sparsity of weight matrices (%)", title="Sparsity", ylim=(0, 100))

    for ax in (ax_loss, ax_acc, ax_sp):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
