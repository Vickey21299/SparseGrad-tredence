"""Mini-batch training and evaluation loops (reused by Parts 3 and 4)."""
import math
import time

import numpy as np
from tqdm import tqdm

from engine import Tensor, backend, no_grad, softmax_cross_entropy
from train.data import iterate_minibatches


def evaluate(model, X, y, batch_size=2000):
    """Mean loss and accuracy over (X, y), without building a graph."""
    total_loss, correct = 0.0, 0
    with no_grad():
        for start in range(0, len(X), batch_size):
            xb, yb = X[start:start + batch_size], y[start:start + batch_size]
            logits = model(Tensor(xb))
            total_loss += softmax_cross_entropy(logits, yb).item() * len(xb)
            correct += int((logits.data.argmax(axis=1) == yb).sum())
    return total_loss / len(X), correct / len(X)


def _step_stats(loss, logits, yb, params):
    """Batch loss, squared global gradient norm and #correct, in ONE device->host copy.

    On the GPU every float()/int() of a device value waits for the GPU to
    finish (a sync). Pulling the three numbers together means one sync per step
    instead of one per parameter plus two.
    """
    xp = backend.xp_of(loss.data)
    grad_sq = sum((p.grad * p.grad).sum() for p in params if p.grad is not None)
    correct = (logits.data.argmax(axis=1) == yb).sum()
    stats = xp.stack([loss.data, grad_sq, correct.astype(loss.data.dtype)])
    loss_value, grad_sq, correct = (float(v) for v in backend.to_numpy(stats))
    return loss_value, grad_sq, int(correct)


def train(model, optimizer, data, epochs, batch_size, rng, logger, log_every=50,
          after_step=None, after_epoch=None, extra_epoch_metrics=None, progress=True):
    """Train with shuffled mini-batches; validate after every epoch.

    after_step(step): called after each optimizer step (Part 3 prunes here).
    after_epoch(record): called with each epoch's metrics (e.g. EarlyStopping);
        training stops if it returns True.
    extra_epoch_metrics(): dict merged into each epoch record.
    progress: show a tqdm progress bar per epoch in the terminal.
    Returns {"epochs": [per-epoch records], "steps": [(step, batch_loss)]}.
    Raises FloatingPointError on the first non-finite loss or gradient.
    """
    X_train, y_train = data["train"]
    params = model.parameters()
    history, step_losses, step = [], [], 0
    batches_per_epoch = math.ceil(len(X_train) / batch_size)

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        loss_sum, correct, seen, max_grad_norm = 0.0, 0, 0, 0.0
        bar = tqdm(iterate_minibatches(X_train, y_train, batch_size, rng), total=batches_per_epoch,
                   desc=f"epoch {epoch}/{epochs}", unit="batch", leave=False, disable=not progress)
        for xb, yb in bar:
            optimizer.zero_grad()
            logits = model(Tensor(xb))
            loss = softmax_cross_entropy(logits, yb)
            loss.backward()
            batch_loss, grad_sq, batch_correct = _step_stats(loss, logits, yb, params)
            if not np.isfinite(batch_loss):
                raise FloatingPointError(f"non-finite loss {batch_loss} at step {step}")
            if not np.isfinite(grad_sq):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            grad_norm = math.sqrt(grad_sq)
            optimizer.step()
            step += 1
            if after_step is not None:
                after_step(step)

            loss_sum += batch_loss * len(xb)
            correct += batch_correct
            seen += len(xb)
            max_grad_norm = max(max_grad_norm, grad_norm)
            step_losses.append((step, batch_loss))
            if step % log_every == 0:
                logger.log_step({"train/batch_loss": batch_loss, "train/grad_norm": grad_norm}, step)
                bar.set_postfix(loss=f"{loss_sum / seen:.4f}", acc=f"{correct / seen:.4f}")
        bar.close()

        val_loss, val_acc = evaluate(model, *data["val"])
        record = {
            "epoch": epoch,
            "step": step,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "max_grad_norm": max_grad_norm,
            "max_abs_weight": max(float(abs(p.data).max()) for p in params),
            "epoch_seconds": time.perf_counter() - start,
        }
        if extra_epoch_metrics is not None:
            record.update(extra_epoch_metrics())
        logger.log_epoch(record)
        history.append(record)
        if after_epoch is not None and after_epoch(record):
            logger.info(f"stopping early after epoch {epoch}")
            break
    return {"epochs": history, "steps": step_losses}


class EarlyStopping:
    """Stop once `monitor` has not improved by `min_delta` for `patience` epochs.

    Also keeps a copy of the weights from the best epoch so far; call
    `restore()` to load them back into the model before testing.
    """

    def __init__(self, model, logger, monitor="val_loss", patience=10, min_delta=1e-4):
        self.model, self.logger = model, logger
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.lower_is_better = monitor.endswith("loss")
        self.best = math.inf if self.lower_is_better else -math.inf
        self.best_epoch, self.best_state, self.wait = 0, None, 0

    def __call__(self, record):
        value = record[self.monitor]
        if self.lower_is_better:
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.wait = value, record["epoch"], 0
            self.best_state = self.model.state_dict()
            self.logger.info(f"new best {self.monitor} {value:.4f} at epoch {record['epoch']}")
        else:
            self.wait += 1
            self.logger.info(f"no improvement in {self.monitor} for {self.wait}/{self.patience} epochs "
                             f"(best {self.best:.4f} at epoch {self.best_epoch})")
        return self.wait >= self.patience

    def restore(self):
        self.model.load_state_dict(self.best_state)
        return self.best_state
