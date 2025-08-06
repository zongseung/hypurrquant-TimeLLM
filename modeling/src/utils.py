# utils.py
import os
import json
import random
import logging
from typing import Any, Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fix random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(log_dir: str, name: str = "TimeLLM") -> logging.Logger:
    """
    Create a logger that writes to both console and a file.

    Args:
        log_dir: directory to save log file
        name: logger name

    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Stream handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def save_args(args: Any, out_path: str) -> None:
    """
    Save argparse Namespace or dict to JSON file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        if hasattr(args, '.__dict__'):
            json.dump(vars(args), f, indent=2)
        else:
            json.dump(args, f, indent=2)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_dir: str
) -> None:
    """
    Save model and optimizer state to checkpoint file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, ckpt_path)


def load_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> int:
    """
    Load model and optimizer state from checkpoint.

    Returns:
        Last epoch number
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    last_epoch = ckpt.get("epoch", 0)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return last_epoch


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Compute common regression metrics between predictions and ground truth.

    Returns:
        Dictionary with MAE, RMSE
    """
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


def ensure_dir(path: str) -> None:
    """
    Create directory if not exists.
    """
    os.makedirs(path, exist_ok=True)