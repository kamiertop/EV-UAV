"""Training run manager — creates run directories, saves configs and checkpoints."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RunManager:
    """Manages a single training run: directory, config save, logging, checkpoints."""

    def __init__(self, root: str | Path = "runs", prefix: str = "train"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root) / f"{prefix}_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        self._log_path = self.run_dir / "metrics.jsonl"
        self._log_file = open(self._log_path, "w")

    # ---- config ----

    def save_config(self, config: dict[str, Any] | object):
        """Save run configuration as JSON. Accepts a dict or an argparse.Namespace."""
        if not isinstance(config, dict):
            config = {k: v for k, v in vars(config).items()
                       if not k.startswith("_")}
        # convert non-serialisable values
        clean: dict[str, Any] = {}
        for k, v in config.items():
            if isinstance(v, Path):
                clean[k] = str(v)
            elif callable(v) or isinstance(v, type):
                continue
            else:
                try:
                    json.dumps({k: v})
                    clean[k] = v
                except (TypeError, ValueError):
                    clean[k] = str(v)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False, default=str)

    # ---- logging ----

    def log_metric(self, epoch: int, batch: int | None, **metrics):
        """Append a line of metrics to the JSONL log."""
        record = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        if batch is not None:
            record["batch"] = batch
        record.update(metrics)
        self._log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log_file.flush()

    # ---- checkpoints ----

    def save_checkpoint(self, model, filename: str):
        import torch
        torch.save(model.state_dict(), self.ckpt_dir / filename)

    @property
    def log_path(self) -> Path:
        return self._log_path

    def close(self):
        self._log_file.close()
