"""Run logging: console + log file + per-epoch CSV, and optionally Weights & Biases."""
import csv
import json
import logging
from pathlib import Path


class RunLogger:
    """Writes everything a run produces into `run_dir`.

    wandb_mode: None to disable W&B, "offline" to record locally (sync later
    with `wandb sync`), or "online" (needs `wandb login` first).
    """

    def __init__(self, run_dir, config, wandb_mode=None, project="self-pruning-network", name=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(config, indent=2))

        self.logger = logging.getLogger(f"run:{self.run_dir.resolve()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        for handler in (logging.StreamHandler(), logging.FileHandler(self.run_dir / "train.log", mode="w")):
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)

        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_path.unlink(missing_ok=True)

        self.wandb_run = None
        if wandb_mode:
            import wandb

            self.wandb_run = wandb.init(project=project, name=name, config=config,
                                        mode=wandb_mode, dir=str(self.run_dir))

    def info(self, message):
        self.logger.info(message)

    def log_step(self, metrics, step):
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def log_epoch(self, record):
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            if new_file:
                writer.writeheader()
            writer.writerow(record)
        self.info(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                             for k, v in record.items()))
        if self.wandb_run is not None:
            self.wandb_run.log({f"epoch/{k}": v for k, v in record.items()}, step=record["step"])

    def log_image(self, key, path):
        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log({key: wandb.Image(str(path))})

    def finish(self, summary=None):
        if summary:
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            self.info("summary: " + json.dumps(summary))
        if self.wandb_run is not None:
            if summary:
                self.wandb_run.summary.update(summary)
            self.wandb_run.finish()
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()
