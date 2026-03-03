"""
VLMSlim — Fine-Tune Vision-Only Teacher Baselines
====================================================
DeiT-B/16 and ResNet-50 are pretrained on ImageNet with 1000-class heads.
They CANNOT be used as KD teachers on CIFAR-100 (100 classes) or CUB-200
(200 classes) without replacing the head and fine-tuning first.

This script fine-tunes them to convergence, saves the checkpoint, and then
cache_teachers.py loads the fine-tuned model to extract logits and features
in the correct label space.

Usage:
    python finetune_teacher.py --teacher deit_vitb16 --dataset cifar100
    python finetune_teacher.py --teacher resnet50_supervised --dataset cifar100
    python finetune_teacher.py --teacher all --dataset cifar100   # Both

The fine-tuned checkpoint is saved to:
    ./finetuned_teachers/{teacher}_{dataset}/best.pth
"""

import argparse
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from config import TEACHERS, DATASETS, TeacherConfig, DatasetConfig
from datasets import (
    CIFAR100_MEAN, CIFAR100_STD, IMAGENET_MEAN, IMAGENET_STD,
    CUB200Dataset,
)


# ──────────────────────────────────────────────────────────
# Per-architecture fine-tuning recipes
# ──────────────────────────────────────────────────────────

FINETUNE_RECIPES = {
    "deit": dict(
        epochs=50, lr=1e-3, weight_decay=0.05,
        optimizer="adamw", warmup_epochs=5, batch_size=128,
    ),
    "resnet50": dict(
        epochs=50, lr=1e-2, weight_decay=1e-4,
        optimizer="sgd", warmup_epochs=5, batch_size=128,
    ),
}


# ──────────────────────────────────────────────────────────
# Checkpoint path helpers  (used by cache_teachers.py too)
# ──────────────────────────────────────────────────────────

def get_finetuned_path(teacher_key: str, dataset_name: str,
                        base_dir: str = "./finetuned_teachers") -> str:
    """Return where a fine-tuned checkpoint should live."""
    return os.path.join(base_dir, f"{teacher_key}_{dataset_name}", "best.pth")


def is_finetuned(teacher_key: str, dataset_name: str,
                  base_dir: str = "./finetuned_teachers") -> bool:
    """Check whether a fine-tuned checkpoint already exists."""
    return os.path.exists(get_finetuned_path(teacher_key, dataset_name, base_dir))


# ──────────────────────────────────────────────────────────
# Model builders (pretrained backbone + fresh head)
# ──────────────────────────────────────────────────────────

def _build_deit(num_classes: int, device: str):
    import timm
    # timm replaces the head automatically when num_classes != 1000
    model = timm.create_model(
        "deit_base_patch16_224", pretrained=True, num_classes=num_classes
    )
    return model.to(device)


def _build_resnet50(num_classes: int, device: str):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    return model.to(device)


MODEL_BUILDERS = {
    "deit": _build_deit,
    "resnet50": _build_resnet50,
}


# ──────────────────────────────────────────────────────────
# Data loaders  (224×224 — teacher resolution)
# ──────────────────────────────────────────────────────────

def get_finetune_loaders(dataset_name: str, data_root: str, batch_size: int = 128):
    """Train / val / test at 224×224 for teacher fine-tuning."""

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if dataset_name == "cifar100":
        full_train = datasets.CIFAR100(
            data_root, train=True, download=True, transform=train_transform)
        full_val = datasets.CIFAR100(
            data_root, train=True, download=True, transform=eval_transform)
        test_set = datasets.CIFAR100(
            data_root, train=False, download=True, transform=eval_transform)
        train_set = Subset(full_train, list(range(45000)))
        val_set = Subset(full_val, list(range(45000, 50000)))

    elif dataset_name == "cub200":
        train_set = CUB200Dataset(data_root, train=True, transform=train_transform)
        val_set = CUB200Dataset(data_root, train=False, transform=eval_transform)
        test_set = val_set
    else:
        raise ValueError(
            f"Fine-tuning on {dataset_name} not supported. "
            f"For ImageNet, use the pretrained model directly."
        )

    kw = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               drop_last=True, **kw)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────
# Core fine-tuning loop
# ──────────────────────────────────────────────────────────

def finetune_teacher(
    teacher_key: str,
    dataset_name: str,
    data_root: str = "./data",
    save_dir: str = "./finetuned_teachers",
    device: str = "cuda",
) -> str:
    """Fine-tune a vision-only teacher to convergence. Returns checkpoint path."""

    teacher_cfg = TEACHERS[teacher_key]
    ds_cfg = DATASETS[dataset_name]
    recipe = FINETUNE_RECIPES[teacher_cfg.model_type]

    save_path = get_finetuned_path(teacher_key, dataset_name, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Skip if already done ──
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        print(f"  [SKIP] {teacher_key} already fine-tuned on {dataset_name}: "
              f"val {ckpt['val_acc']:.2f}%, test {ckpt['test_acc']:.2f}%")
        return save_path

    print(f"\n{'='*70}")
    print(f"  FINE-TUNING: {teacher_key} on {dataset_name}")
    print(f"  Epochs: {recipe['epochs']}, LR: {recipe['lr']}, "
          f"Optimizer: {recipe['optimizer']}")
    print(f"{'='*70}\n")

    # ── Model ──
    builder = MODEL_BUILDERS[teacher_cfg.model_type]
    model = builder(ds_cfg.num_classes, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {n_params:.1f}M params")

    # ── Data ──
    train_loader, val_loader, test_loader = get_finetune_loaders(
        dataset_name, data_root, recipe["batch_size"]
    )
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # ── Optimizer + scheduler ──
    if recipe["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=recipe["lr"],
            weight_decay=recipe["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=recipe["lr"],
            momentum=0.9, weight_decay=recipe["weight_decay"],
        )

    warmup = recipe["warmup_epochs"]
    total = recipe["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / (total - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # ── Train ──
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(total):
        # ── Train epoch ──
        model.train()
        running_loss = 0.0
        correct = 0
        n_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            n_total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc=f"{100.*correct/n_total:.1f}%")

        scheduler.step()
        train_acc = 100.0 * correct / n_total

        # ── Validate ──
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += model(images).argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if (epoch + 1) % 10 == 0 or epoch == total - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:>3d}/{total} | "
                  f"Train {train_acc:.1f}% | "
                  f"Val {val_acc:.1f}% (best {best_val_acc:.1f}% @ {best_epoch+1}) | "
                  f"LR {lr:.6f}")

    # ── Final test ──
    model.load_state_dict(best_state)
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test_correct += model(images).argmax(1).eq(labels).sum().item()
            test_total += labels.size(0)
    test_acc = 100.0 * test_correct / test_total

    # ── Save ──
    torch.save({
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "teacher_key": teacher_key,
        "dataset": dataset_name,
        "num_classes": ds_cfg.num_classes,
        "model_type": teacher_cfg.model_type,
    }, save_path)

    print(f"\n  ✓ Saved fine-tuned {teacher_key} → {save_path}")
    print(f"    Val: {best_val_acc:.2f}% (epoch {best_epoch+1})")
    print(f"    Test: {test_acc:.2f}%")
    print(f"    Ready for: python cache_teachers.py --dataset {dataset_name}\n")
    return save_path


# ──────────────────────────────────────────────────────────
# Loader used by cache_teachers.py
# ──────────────────────────────────────────────────────────

def load_finetuned_model(teacher_key: str, dataset_name: str,
                          device: str = "cuda",
                          save_dir: str = "./finetuned_teachers"):
    """Load a fine-tuned vision-only teacher.

    Returns: (model, num_classes)
    """
    teacher_cfg = TEACHERS[teacher_key]
    ckpt_path = get_finetuned_path(teacher_key, dataset_name, save_dir)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {ckpt_path}\n"
            f"Run: python finetune_teacher.py "
            f"--teacher {teacher_key} --dataset {dataset_name}"
        )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]

    # Rebuild the same architecture with the correct num_classes
    if teacher_cfg.model_type == "deit":
        import timm
        model = timm.create_model(
            "deit_base_patch16_224", pretrained=False, num_classes=num_classes
        )
    elif teacher_cfg.model_type == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {teacher_cfg.model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    print(f"  Loaded fine-tuned {teacher_key} on {dataset_name}: "
          f"val={ckpt['val_acc']:.1f}%, test={ckpt['test_acc']:.1f}%")
    return model, num_classes


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune vision-only teachers for VLMSlim Exp 0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune_teacher.py --teacher deit_vitb16 --dataset cifar100
  python finetune_teacher.py --teacher resnet50_supervised --dataset cifar100
  python finetune_teacher.py --teacher all --dataset cifar100
        """,
    )
    parser.add_argument("--teacher", type=str, required=True,
                        help="Teacher key or 'all' for both vision-only teachers")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar100", "cub200"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./finetuned_teachers")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    DATASETS[args.dataset].data_root = args.data_root

    if args.teacher == "all":
        teachers = [k for k, v in TEACHERS.items() if v.requires_finetune]
    else:
        teachers = [args.teacher]

    for key in teachers:
        if key not in TEACHERS:
            print(f"  [ERROR] Unknown teacher: {key}")
            continue
        if not TEACHERS[key].requires_finetune:
            print(f"  [SKIP] {key} is a VLM teacher — no fine-tuning needed.")
            continue
        finetune_teacher(
            teacher_key=key, dataset_name=args.dataset,
            data_root=args.data_root, save_dir=args.save_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
