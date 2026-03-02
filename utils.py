"""
VLMSlim — Utilities
====================
Logging, metrics, teacher scoring, reproducibility.
"""

import os
import json
import time
import random
import numpy as np
import torch
from typing import Dict, List, Optional
from collections import defaultdict


# ──────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────
# Teacher scoring
# ──────────────────────────────────────────────────────────

def load_teacher_scores(cache_dir: str, dataset_name: str,
                         teacher_names: List[str]) -> Dict[str, float]:
    """Load pre-computed zero-shot scores for teachers.

    Returns: {teacher_name: accuracy} sorted best→worst
    """
    scores = {}
    split = "val" if dataset_name in ("cifar100", "imagenet") else "test"
    save_dir = os.path.join(cache_dir, dataset_name, split)

    for name in teacher_names:
        score_path = os.path.join(save_dir, f"{name}_score.txt")
        if os.path.exists(score_path):
            with open(score_path, "r") as f:
                scores[name] = float(f.read().strip())
        else:
            # Default: use arbitrary ordering if scores not available
            print(f"  [WARN] No cached score for {name}, using 50.0 as default")
            scores[name] = 50.0

    return scores


def order_teachers_by_score(scores: Dict[str, float]) -> List[str]:
    """Return teacher names ordered best→worst (highest score first)."""
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def compute_feature_metrics(features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute inter-class / intra-class distance ratio for cross-modal awareness analysis.

    Higher ratio = better class separation in feature space.
    """
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"distance_ratio": 0.0}

    # Compute class centroids
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = features[mask].mean(axis=0)

    # Intra-class: average distance of samples to their class centroid
    intra_dists = []
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        centroid = centroids[label]
        dists = np.linalg.norm(class_features - centroid, axis=1)
        intra_dists.extend(dists.tolist())
    mean_intra = np.mean(intra_dists) if intra_dists else 1e-8

    # Inter-class: average distance between class centroids
    centroid_array = np.stack(list(centroids.values()))
    n_classes = len(centroid_array)
    inter_dists = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            inter_dists.append(np.linalg.norm(centroid_array[i] - centroid_array[j]))
    mean_inter = np.mean(inter_dists) if inter_dists else 0.0

    ratio = mean_inter / max(mean_intra, 1e-8)
    return {
        "inter_class_dist": float(mean_inter),
        "intra_class_dist": float(mean_intra),
        "distance_ratio": float(ratio),
    }


# ──────────────────────────────────────────────────────────
# Experiment Logger
# ──────────────────────────────────────────────────────────

class ExperimentLogger:
    """Logs all metrics from the logging checklist to JSON + optional wandb.

    Logged per epoch:
        - val_acc, test_acc
        - loss_total, loss_ce, loss_kd, loss_feat, loss_anchor
        - grad_norm
        - anchor_magnitude
        - gamma_value (derived at epoch 1)
        - wall_clock_seconds
        - phase_idx, teacher_name
    """

    def __init__(self, output_dir: str, exp_name: str, use_wandb: bool = False,
                 wandb_project: str = "vlmslim"):
        self.output_dir = output_dir
        self.exp_name = exp_name
        self.use_wandb = use_wandb
        self.history = defaultdict(list)
        self.metadata = {}

        os.makedirs(output_dir, exist_ok=True)

        if use_wandb:
            import wandb
            wandb.init(project=wandb_project, name=exp_name,
                       dir=output_dir, reinit=True)

    def log_metadata(self, key: str, value):
        """Log static metadata (seed, config, derived gamma, etc.)"""
        self.metadata[key] = value

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for a single epoch."""
        metrics["epoch"] = epoch
        for k, v in metrics.items():
            self.history[k].append(v)

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=epoch)

    def log_phase_boundary(self, epoch: int, phase_idx: int, teacher_name: str,
                            snapshot_time: float):
        """Log phase transition event."""
        event = {
            "epoch": epoch,
            "phase_idx": phase_idx,
            "teacher_name": teacher_name,
            "snapshot_time_seconds": snapshot_time,
        }
        if "phase_events" not in self.metadata:
            self.metadata["phase_events"] = []
        self.metadata["phase_events"].append(event)

    def save(self):
        """Save full history to JSON."""
        output = {
            "metadata": self.metadata,
            "history": dict(self.history),
        }
        path = os.path.join(self.output_dir, "experiment_log.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  [Log] Saved to {path}")

        if self.use_wandb:
            import wandb
            wandb.finish()

    def get_best_val_epoch(self) -> int:
        """Return epoch with highest validation accuracy."""
        if "val_acc" not in self.history:
            return -1
        accs = self.history["val_acc"]
        return int(np.argmax(accs))


# ──────────────────────────────────────────────────────────
# Gradient norm computation
# ──────────────────────────────────────────────────────────

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ──────────────────────────────────────────────────────────
# Learning rate scheduler with warmup
# ──────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup + cosine annealing scheduler.

    Warmup: linearly increase LR from warmup_start_lr to base_lr over warmup_epochs.
    Cosine: decay from base_lr to 0 over remaining epochs.
    """

    def __init__(self, optimizer, base_lr: float, total_epochs: int,
                 warmup_epochs: int = 5, warmup_start_lr: float = None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr or (base_lr * 0.01)

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (
                epoch / self.warmup_epochs
            )
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = 0.5 * self.base_lr * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
