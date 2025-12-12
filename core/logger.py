"""
A small experiment logging helper:
- Wraps torch.utils.tensorboard.SummaryWriter
- Keeps lists of scalars (episode reward / loss) in memory for export
- Save and load model helper
- Export matplotlib PNGs for reward/loss curves to experiments/graphs/
"""

from typing import Optional, Dict, Any, List
import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class RLLogger:
    """
    RLLogger wraps SummaryWriter and collects episode-level scalars for easy export.
    Usage:
        logger = RLLogger(log_dir)
        logger.log_scalar("episode_reward", reward, step)
        logger.log_info({"lr": 1e-4}, step)
        logger.save_model(policy_net, "policy_latest.pt")
        logger.export_graphs(out_dir="experiments/graphs")
    """
    def __init__(self, log_dir: str, config: Optional[Dict[str, Any]] = None):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        # in-memory time series for quick plotting/export
        self.storage: Dict[str, List[float]] = {}
        os.makedirs(self.log_dir, exist_ok=True)
        if config:
            with open(os.path.join(self.log_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar to both TensorBoard and in-memory storage.
        step: if None, will use length of series as step.
        """
        if step is None:
            step = len(self.storage.get(name, []))
        self.writer.add_scalar(name, value, step)
        self.storage.setdefault(name, []).append(float(value))

    def log_info(self, info: Dict[str, Any], step: Optional[int] = None):
        """Log a dict of scalars."""
        for k, v in info.items():
            try:
                self.log_scalar(k, float(v), step)
            except Exception:
                # skip non-float entries
                pass

    def save_model(self, model: torch.nn.Module, filename: str = "model.pt"):
        """Save a PyTorch model in the log directory."""
        path = os.path.join(self.log_dir, filename)
        torch.save(model.state_dict(), path)

    def save_checkpoint(self, state: Dict[str, Any], filename: str = "checkpoint.pt"):
        path = os.path.join(self.log_dir, filename)
        torch.save(state, path)

    def export_graphs(self, out_dir: str = "experiments/graphs"):
        """Export stored series as PNG plots (reward/loss/etc)."""
        os.makedirs(out_dir, exist_ok=True)
        for name, series in self.storage.items():
            if not series:
                continue
            values = np.array(series, dtype=float)
            plt.figure(figsize=(6, 4))
            plt.plot(values)
            plt.xlabel("step")
            plt.ylabel(name)
            plt.title(name)
            plt.grid(alpha=0.3)
            fname = f"{name.replace('/', '_')}.png"
            out_path = os.path.join(out_dir, fname)
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()

    def close(self):
        self.writer.close()