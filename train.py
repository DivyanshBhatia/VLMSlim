"""
VLMSlim — Training Loop
========================
Core training engine with:
  - Phase-aware sequential teacher scheduling
  - Cumulative target construction
  - Anchor snapshotting at phase boundaries
  - Automatic γ derivation
  - Complete per-epoch logging (all 12 checklist items)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional

from config import ExperimentConfig, TEACHERS
from models import StudentModel, ProjectionAdaptor, load_student
from losses import VLMSlimLoss
from utils import (
    set_seed, accuracy, compute_gradient_norm, compute_feature_metrics,
    ExperimentLogger, WarmupCosineScheduler,
    load_teacher_scores, order_teachers_by_score,
)


def train_one_epoch(
    model: StudentModel,
    adaptors: Dict[str, ProjectionAdaptor],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: VLMSlimLoss,
    active_teacher: str,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch. Returns averaged loss components."""

    model.train()
    for adaptor in adaptors.values():
        adaptor.train()

    running = {"total": 0, "ce": 0, "kd": 0, "feat": 0, "anchor": 0, "grad_norm": 0}
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        if len(batch) == 3:
            images, labels, teacher_data = batch
        else:
            images, labels = batch
            teacher_data = None

        images = images.to(device)
        labels = labels.to(device)

        # Student forward
        student_logits, student_features = model(images)

        # Collect teacher logits for active teachers
        teacher_logits = {}
        projected_teacher_feat = None

        if teacher_data is not None:
            for t_name in loss_fn.target_builder.active_teachers:
                if t_name in teacher_data:
                    td = teacher_data[t_name]
                    if "logits" in td:
                        teacher_logits[t_name] = td["logits"].to(device)

            # Feature alignment: project the CURRENT phase's primary teacher features
            if loss_fn.use_feature and active_teacher in teacher_data:
                td = teacher_data[active_teacher]
                if "features" in td and active_teacher in adaptors:
                    t_feat = td["features"].to(device)
                    projected_teacher_feat = adaptors[active_teacher](t_feat)

        # Compute loss
        if not teacher_logits:
            # Fallback: pure CE training (shouldn't happen in normal flow)
            loss = F.cross_entropy(student_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running["total"] += loss.item()
            running["ce"] += loss.item()
            n_batches += 1
            continue

        losses = loss_fn(
            student_logits=student_logits,
            student_features=student_features,
            labels=labels,
            teacher_logits=teacher_logits,
            projected_teacher_features=projected_teacher_feat,
            model=model,
            batch_idx=batch_idx,
        )

        # Backward + step
        optimizer.zero_grad()
        losses["total"].backward()

        grad_norm = compute_gradient_norm(model)
        optimizer.step()

        # Accumulate
        running["total"] += losses["total"].item()
        running["ce"] += losses["ce"].item()
        running["kd"] += losses["kd"].item()
        running["feat"] += losses["feat"].item()
        running["anchor"] += losses["anchor"].item()
        running["grad_norm"] += grad_norm
        n_batches += 1

    # Average
    for k in running:
        running[k] /= max(n_batches, 1)

    running["gamma_value"] = loss_fn.gamma if loss_fn.gamma is not None else 50.0
    return running


@torch.no_grad()
def evaluate(
    model: StudentModel,
    loader: DataLoader,
    device: str,
    collect_features: bool = False,
) -> Dict[str, float]:
    """Evaluate on val or test set. Returns accuracy and optional feature metrics."""

    model.eval()
    correct = 0
    total = 0
    all_features = []
    all_labels = []

    for batch in loader:
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        logits, features = model(images)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if collect_features:
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    result = {"acc": 100.0 * correct / total}

    if collect_features and all_features:
        features_cat = torch.cat(all_features)
        labels_cat = torch.cat(all_labels)
        feat_metrics = compute_feature_metrics(features_cat, labels_cat)
        result.update(feat_metrics)

    return result


def run_experiment(cfg: ExperimentConfig):
    """Execute a complete VLMSlim experiment.

    This function handles the full training pipeline:
    1. Load data with cached teacher outputs
    2. Initialize student + projection adaptors
    3. Set up phase schedule from teacher scores
    4. Train with phase transitions
    5. Evaluate and log everything

    Returns: path to experiment log JSON
    """
    from datasets import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    ds_cfg = cfg.get_dataset_config()
    student_cfg = cfg.get_student_config()
    teacher_cfgs = cfg.get_teacher_configs()
    output_path = cfg.get_output_path()

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {cfg.exp_name}")
    print(f"  ID:         {cfg.exp_id}")
    print(f"  Dataset:    {cfg.dataset} | Student: {cfg.student} | Seed: {cfg.seed}")
    print(f"  Teachers:   {cfg.teachers}")
    print(f"  Switches:   cumulative={cfg.use_cumulative_targets}, "
          f"anchor={cfg.use_anchor}, feature={cfg.use_feature_path}, "
          f"sequential={cfg.sequential}")
    print(f"  λ={cfg.lam}, α={cfg.alpha}, β={cfg.beta}, τ={cfg.tau}")
    print(f"  Output:     {output_path}")
    print(f"{'='*70}\n")

    # ── 1. Load teacher scores and order ──
    teacher_scores = load_teacher_scores(cfg.cache_dir, cfg.dataset, cfg.teachers)
    teacher_order = order_teachers_by_score(teacher_scores)

    print(f"  Teacher scores (zero-shot on {cfg.dataset}):")
    for t in teacher_order:
        print(f"    {t}: {teacher_scores[t]:.2f}%")

    # For non-sequential experiments, use all teachers from epoch 1
    if not cfg.sequential:
        teacher_order = cfg.teachers  # Use original order

    # ── 2. Load data ──
    print(f"\n  Loading dataset: {cfg.dataset}")
    ds_cfg.data_root = ds_cfg.data_root  # Use default or override
    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.dataset, ds_cfg,
        cache_dir=cfg.cache_dir,
        teacher_names=cfg.teachers,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── 3. Initialize student ──
    input_size = ds_cfg.train_size[0]
    student = load_student(student_cfg, ds_cfg.num_classes, input_size).to(device)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {student_cfg.name} ({total_params/1e6:.1f}M params)")

    # ── 4. Initialize projection adaptors (one per teacher) ──
    adaptors = {}
    if cfg.use_feature_path:
        for t_name in cfg.teachers:
            t_cfg = TEACHERS[t_name]
            adaptor = ProjectionAdaptor(t_cfg.feature_dim, student_cfg.feature_dim).to(device)
            adaptors[t_name] = adaptor

    # ── 5. Optimizer (student + all adaptors) ──
    all_params = list(student.parameters())
    for adaptor in adaptors.values():
        all_params += list(adaptor.parameters())

    optimizer = torch.optim.SGD(
        all_params,
        lr=ds_cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, base_lr=ds_cfg.lr,
        total_epochs=ds_cfg.total_epochs,
        warmup_epochs=cfg.warmup_epochs,
    )

    # ── 6. Loss function ──
    loss_fn = VLMSlimLoss(
        alpha=cfg.alpha,
        beta=cfg.beta,
        tau=cfg.tau,
        lam=cfg.lam,
        teacher_scores=teacher_scores,
        use_cumulative=cfg.use_cumulative_targets,
        use_anchor=cfg.use_anchor,
        use_feature=cfg.use_feature_path,
    )

    # ── 7. Phase schedule ──
    if cfg.sequential:
        phase_boundaries = cfg.get_phase_boundaries()
        # Phase i runs from boundary[i-1]+1 to boundary[i]
        # Phase 0 starts at epoch 0
        phases = []
        start = 0
        for i, teacher_name in enumerate(teacher_order):
            end = phase_boundaries[i] if i < len(phase_boundaries) else ds_cfg.total_epochs
            phases.append({
                "teacher": teacher_name,
                "start_epoch": start,
                "end_epoch": end,
                "phase_idx": i,
            })
            start = end
        print(f"\n  Phase schedule (sequential):")
        for p in phases:
            print(f"    Phase {p['phase_idx']}: epochs {p['start_epoch']}–{p['end_epoch']} "
                  f"→ {p['teacher']}")
    else:
        # Non-sequential: all teachers active from epoch 0
        phases = [{
            "teacher": None,  # All active
            "start_epoch": 0,
            "end_epoch": ds_cfg.total_epochs,
            "phase_idx": 0,
        }]
        # Activate all teachers immediately
        for t in teacher_order:
            loss_fn.target_builder.add_teacher(t)
        print(f"\n  Non-sequential: all teachers active from epoch 0")

    # ── 8. Logger ──
    logger = ExperimentLogger(
        output_dir=output_path,
        exp_name=cfg.exp_name,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
    )
    logger.log_metadata("config", {
        "exp_name": cfg.exp_name, "exp_id": cfg.exp_id, "seed": cfg.seed,
        "dataset": cfg.dataset, "student": cfg.student,
        "teachers": cfg.teachers, "teacher_scores": teacher_scores,
        "alpha": cfg.alpha, "beta": cfg.beta, "tau": cfg.tau, "lam": cfg.lam,
        "use_cumulative": cfg.use_cumulative_targets,
        "use_anchor": cfg.use_anchor, "use_feature": cfg.use_feature_path,
        "sequential": cfg.sequential,
        "phase_boundaries": cfg.get_phase_boundaries() if cfg.sequential else [],
        "total_epochs": ds_cfg.total_epochs, "lr": ds_cfg.lr,
        "batch_size": ds_cfg.batch_size,
    })

    # ── 9. Training loop ──
    best_val_acc = 0.0
    best_epoch = 0
    current_phase_idx = -1
    active_teacher = teacher_order[0] if cfg.sequential else teacher_order[0]

    peak_memory = 0
    total_start_time = time.time()

    for epoch in range(ds_cfg.total_epochs):
        epoch_start = time.time()

        # ── Phase transition check ──
        if cfg.sequential:
            for phase in phases:
                if phase["start_epoch"] == epoch and phase["phase_idx"] != current_phase_idx:
                    current_phase_idx = phase["phase_idx"]
                    active_teacher = phase["teacher"]

                    # Snapshot + teacher activation
                    snap_start = time.time()
                    loss_fn.begin_phase(active_teacher, student, current_phase_idx)
                    snap_time = time.time() - snap_start

                    logger.log_phase_boundary(
                        epoch, current_phase_idx, active_teacher, snap_time
                    )
                    print(f"\n  ──── Phase {current_phase_idx} ────"
                          f" Teacher: {active_teacher} "
                          f"(snapshot: {snap_time:.2f}s)")
                    break

        # ── LR schedule ──
        lr = scheduler.step(epoch)

        # ── Train ──
        train_metrics = train_one_epoch(
            model=student,
            adaptors=adaptors,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            active_teacher=active_teacher,
            device=device,
            epoch=epoch,
        )

        # ── Validate ──
        collect_feats = (epoch == ds_cfg.total_epochs - 1)  # Features only at last epoch
        val_result = evaluate(student, val_loader, device, collect_features=collect_feats)
        val_acc = val_result["acc"]

        # ── Track best ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best checkpoint
            ckpt_path = os.path.join(output_path, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "student_state_dict": student.state_dict(),
                "adaptor_state_dicts": {k: v.state_dict() for k, v in adaptors.items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)

        # ── GPU memory ──
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            peak_memory = max(peak_memory, mem)

        epoch_time = time.time() - epoch_start

        # ── Anchor magnitude ──
        anchor_mag = 0.0
        if cfg.use_anchor and loss_fn.anchor_loss.snapshot is not None:
            with torch.no_grad():
                for name, param in student.named_parameters():
                    if param.requires_grad and name in loss_fn.anchor_loss.snapshot:
                        anchor_mag += ((param - loss_fn.anchor_loss.snapshot[name]) ** 2).sum().item()

        # ── Log everything ──
        epoch_metrics = {
            "val_acc": val_acc,
            "loss_total": train_metrics["total"],
            "loss_ce": train_metrics["ce"],
            "loss_kd": train_metrics["kd"],
            "loss_feat": train_metrics["feat"],
            "loss_anchor": train_metrics["anchor"],
            "grad_norm": train_metrics["grad_norm"],
            "anchor_magnitude": anchor_mag,
            "gamma_value": train_metrics["gamma_value"],
            "lr": lr,
            "wall_clock_seconds": epoch_time,
            "phase_idx": current_phase_idx if cfg.sequential else 0,
            "active_teacher": active_teacher,
        }

        # Feature metrics at last epoch
        if collect_feats:
            for k, v in val_result.items():
                if k != "acc":
                    epoch_metrics[f"feature_{k}"] = v

        logger.log_epoch(epoch, epoch_metrics)

        # ── Print progress ──
        if epoch % 10 == 0 or epoch == ds_cfg.total_epochs - 1:
            print(f"  Epoch {epoch:>3d}/{ds_cfg.total_epochs} | "
                  f"Val {val_acc:.2f}% (best: {best_val_acc:.2f}% @ {best_epoch}) | "
                  f"Loss {train_metrics['total']:.4f} "
                  f"(CE:{train_metrics['ce']:.3f} KD:{train_metrics['kd']:.3f} "
                  f"F:{train_metrics['feat']:.3f} A:{train_metrics['anchor']:.3f}) | "
                  f"GradN:{train_metrics['grad_norm']:.2f} | "
                  f"LR:{lr:.5f} | {epoch_time:.1f}s")

    # ── 10. Final test evaluation ──
    print(f"\n  Loading best model from epoch {best_epoch}...")
    ckpt = torch.load(os.path.join(output_path, "best_model.pth"),
                       map_location=device, weights_only=False)
    student.load_state_dict(ckpt["student_state_dict"])

    test_result = evaluate(student, test_loader, device, collect_features=True)
    test_acc = test_result["acc"]

    total_time = time.time() - total_start_time

    # ── Save final metadata ──
    logger.log_metadata("final_results", {
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "total_training_time_seconds": total_time,
        "peak_gpu_memory_gb": peak_memory,
        "feature_metrics": {k: v for k, v in test_result.items() if k != "acc"},
    })

    if loss_fn.gamma is not None:
        logger.log_metadata("derived_gamma", loss_fn.gamma)

    logger.save()

    print(f"\n{'='*70}")
    print(f"  FINISHED: {cfg.exp_name}")
    print(f"  Test Accuracy:     {test_acc:.2f}%")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Total Time:        {total_time/60:.1f} min")
    print(f"  Peak GPU Memory:   {peak_memory:.2f} GB")
    if loss_fn.gamma is not None:
        print(f"  Derived γ:         {loss_fn.gamma:.4f}")
    print(f"  Log:               {output_path}/experiment_log.json")
    print(f"{'='*70}\n")

    return os.path.join(output_path, "experiment_log.json")
