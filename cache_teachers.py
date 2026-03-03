"""
VLMSlim — Teacher Output Caching
==================================
Pre-compute and save teacher logits + features to disk.
This is run ONCE per dataset before any experiments.

For VLM teachers (CLIP, OpenCLIP, MetaCLIP): uses zero-shot classification
via text embeddings — no fine-tuning needed.

For vision-only teachers (DeiT, ResNet-50): loads a FINE-TUNED checkpoint
with the correct number of output classes. You must run finetune_teacher.py
BEFORE caching these teachers. The raw pretrained models output 1000 ImageNet
logits, which is wrong for CIFAR-100 (100 classes) or CUB-200 (200 classes).

Usage:
    # Step 1: Fine-tune vision-only baselines (only needed for Exp 0)
    python finetune_teacher.py --teacher all --dataset cifar100

    # Step 2: Cache all teacher outputs
    python cache_teachers.py --dataset cifar100 --data_root ./data --cache_dir ./cache

    # Cache only VLM teachers (skip vision-only if not yet fine-tuned)
    python cache_teachers.py --dataset cifar100 \
        --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from config import TEACHERS, DATASETS, TeacherConfig
from datasets import (
    CIFAR100_MEAN, CIFAR100_STD, IMAGENET_MEAN, IMAGENET_STD,
    CUB200Dataset,
)


# ──────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────

def get_teacher_transform(teacher_cfg: TeacherConfig, teacher_size: int = 224):
    """Get the appropriate preprocessing transform for a teacher."""
    return transforms.Compose([
        transforms.Resize((teacher_size, teacher_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_templates(dataset_name: str):
    """Get text prompt templates appropriate for each dataset."""
    if dataset_name == "cub200":
        return ["a photo of a {}, a type of bird."]
    return [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a photo of the large {}.",
        "a photo of the small {}.",
        "a photo of a {}, a type of object.",
    ]


def get_dataset_and_classnames(dataset_name, split, data_root, teacher_transform):
    """Load dataset with teacher-sized transforms. Returns (dataset, class_names)."""

    if dataset_name == "cifar100":
        is_train = (split in ("train", "val"))
        raw_dataset = datasets.CIFAR100(data_root, train=is_train, download=True,
                                         transform=teacher_transform)
        if split == "train":
            raw_dataset = Subset(raw_dataset, list(range(45000)))
        elif split == "val":
            raw_dataset = Subset(raw_dataset, list(range(45000, 50000)))
        class_names = (raw_dataset.dataset.classes
                       if isinstance(raw_dataset, Subset) else raw_dataset.classes)

    elif dataset_name == "cub200":
        is_train = (split in ("train", "val"))
        raw_dataset = CUB200Dataset(data_root, train=is_train, transform=teacher_transform)
        class_names = CUB200Dataset.get_class_names(data_root)

    elif dataset_name == "imagenet":
        folder = "train" if split == "train" else "val"
        raw_dataset = datasets.ImageFolder(
            os.path.join(data_root, "imagenet", folder),
            transform=teacher_transform,
        )
        class_names = [name.replace("_", " ") for name in raw_dataset.classes]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return raw_dataset, class_names


def collect_labels(loader):
    """Iterate through loader once to collect all labels."""
    all_labels = []
    for _, labels in loader:
        all_labels.append(labels)
    return torch.cat(all_labels)


def compute_and_save_accuracy(all_logits, loader, save_dir, teacher_key, split):
    """Compute accuracy on cached logits, print, and write score file."""
    all_labels = collect_labels(loader)
    preds = all_logits.argmax(dim=1)
    correct = (preds == all_labels).sum().item()
    acc = 100.0 * correct / len(all_labels)
    print(f"  Zero-shot accuracy ({split}): {acc:.2f}%")

    score_path = os.path.join(save_dir, f"{teacher_key}_score.txt")
    with open(score_path, "w") as f:
        f.write(f"{acc:.4f}")
    return acc


def build_zero_shot_classifier(model, tokenizer, class_names, templates, device,
                                 model_type="openclip"):
    """Build a zero-shot classification weight matrix from text embeddings.

    Returns: (feature_dim, num_classes) weight matrix
    """
    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(class_names, desc="Building zero-shot classifier"):
            texts = [t.format(classname) for t in templates]

            if model_type in ("clip", "openclip"):
                tokens = tokenizer(texts).to(device)
                text_features = model.encode_text(tokens)
            elif model_type == "siglip":
                inputs = tokenizer(text=texts, padding="max_length", truncation=True,
                                    return_tensors="pt").to(device)
                text_features = model.text_model(
                    **{k: v for k, v in inputs.items() if k != "pixel_values"}
                ).pooler_output
            else:
                raise ValueError(f"Cannot build zero-shot classifier for {model_type}")

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            zeroshot_weights.append(text_features)

    return torch.stack(zeroshot_weights, dim=1)  # (feat_dim, num_classes)


# ──────────────────────────────────────────────────────────
# VLM Teacher Caching  (zero-shot — no fine-tuning)
# ──────────────────────────────────────────────────────────

def cache_vlm_teacher(teacher_key, teacher_cfg, dataset_name, split,
                       data_root, cache_dir, device, batch_size):
    """Cache logits + features for a VLM teacher via zero-shot classification."""

    ds_cfg = DATASETS[dataset_name]
    save_dir = os.path.join(cache_dir, dataset_name, split)
    os.makedirs(save_dir, exist_ok=True)

    logit_path = os.path.join(save_dir, f"{teacher_key}_logits.pt")
    feat_path = os.path.join(save_dir, f"{teacher_key}_features.pt")

    if os.path.exists(logit_path) and os.path.exists(feat_path):
        print(f"  [SKIP] {teacher_key}/{dataset_name}/{split} already cached.")
        return

    print(f"\n{'='*60}")
    print(f"  Caching VLM teacher: {teacher_key} on {dataset_name}/{split}")
    print(f"{'='*60}")

    teacher_transform = get_teacher_transform(teacher_cfg, teacher_size=224)
    raw_dataset, class_names = get_dataset_and_classnames(
        dataset_name, split, data_root, teacher_transform
    )
    loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    templates = get_templates(dataset_name)

    # ── Load model + build zero-shot head ──
    if teacher_cfg.model_type in ("clip", "openclip"):
        import open_clip
        model_name, pretrained = teacher_cfg.checkpoint.split("/")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        zs_weights = build_zero_shot_classifier(
            model, tokenizer, class_names, templates, device,
            model_type=teacher_cfg.model_type,
        )

        @torch.no_grad()
        def extract(images):
            features = model.encode_image(images)
            features_normed = features / features.norm(dim=-1, keepdim=True)
            logits = 100.0 * features_normed @ zs_weights
            return logits, features

    elif teacher_cfg.model_type == "siglip":
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(teacher_cfg.checkpoint).to(device).eval()
        processor = AutoProcessor.from_pretrained(teacher_cfg.checkpoint)

        zs_weights = build_zero_shot_classifier(
            model, processor, class_names, templates, device, model_type="siglip"
        )

        @torch.no_grad()
        def extract(images):
            outputs = model.vision_model(pixel_values=images)
            features = outputs.pooler_output
            features_normed = features / features.norm(dim=-1, keepdim=True)
            logits = features_normed @ zs_weights
            return logits, features

    else:
        raise ValueError(f"Unknown VLM teacher type: {teacher_cfg.model_type}")

    # ── Extract ──
    all_logits = []
    all_features = []
    for images, _ in tqdm(loader, desc=f"Extracting {teacher_key}"):
        images = images.to(device)
        logits, features = extract(images)
        all_logits.append(logits.cpu())
        all_features.append(features.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_features = torch.cat(all_features, dim=0)

    # ── Dimension sanity check ──
    assert all_logits.shape[1] == ds_cfg.num_classes, (
        f"Logit dim mismatch for {teacher_key}: got {all_logits.shape[1]}, "
        f"expected {ds_cfg.num_classes} for {dataset_name}"
    )

    torch.save(all_logits, logit_path)
    torch.save(all_features, feat_path)
    print(f"  Saved logits:   {all_logits.shape} → {logit_path}")
    print(f"  Saved features: {all_features.shape} → {feat_path}")

    if split == "val" or (split == "test" and dataset_name != "imagenet"):
        compute_and_save_accuracy(all_logits, loader, save_dir, teacher_key, split)


# ──────────────────────────────────────────────────────────
# Vision-Only Teacher Caching  (requires fine-tuned checkpoint)
# ──────────────────────────────────────────────────────────

def cache_vision_teacher(teacher_key, teacher_cfg, dataset_name, split,
                          data_root, cache_dir, device, batch_size,
                          finetuned_dir="./finetuned_teachers"):
    """Cache logits + features for a fine-tuned vision-only teacher.

    The fine-tuned checkpoint MUST match the target dataset's num_classes.
    Without fine-tuning, DeiT outputs 1000-class ImageNet logits and
    ResNet-50 outputs 1000-class logits — both wrong for CIFAR-100/CUB-200.
    """
    from finetune_teacher import load_finetuned_model, is_finetuned

    ds_cfg = DATASETS[dataset_name]
    save_dir = os.path.join(cache_dir, dataset_name, split)
    os.makedirs(save_dir, exist_ok=True)

    logit_path = os.path.join(save_dir, f"{teacher_key}_logits.pt")
    feat_path = os.path.join(save_dir, f"{teacher_key}_features.pt")

    if os.path.exists(logit_path) and os.path.exists(feat_path):
        print(f"  [SKIP] {teacher_key}/{dataset_name}/{split} already cached.")
        return

    # ── Guard: fine-tuned checkpoint must exist ──
    if not is_finetuned(teacher_key, dataset_name, finetuned_dir):
        print(f"\n  ⚠  CANNOT CACHE {teacher_key} on {dataset_name}:")
        print(f"     Vision-only teachers must be fine-tuned on the target dataset")
        print(f"     before caching. The raw pretrained model outputs 1000 ImageNet")
        print(f"     logits, but {dataset_name} has {ds_cfg.num_classes} classes.")
        print(f"")
        print(f"     Fix: python finetune_teacher.py "
              f"--teacher {teacher_key} --dataset {dataset_name}")
        print()
        return

    print(f"\n{'='*60}")
    print(f"  Caching vision-only teacher: {teacher_key} on {dataset_name}/{split}")
    print(f"  (using fine-tuned checkpoint)")
    print(f"{'='*60}")

    model, num_classes = load_finetuned_model(
        teacher_key, dataset_name, device, finetuned_dir
    )
    assert num_classes == ds_cfg.num_classes

    teacher_transform = get_teacher_transform(teacher_cfg, teacher_size=224)
    raw_dataset, _ = get_dataset_and_classnames(
        dataset_name, split, data_root, teacher_transform
    )
    loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # ── Build extraction function ──
    if teacher_cfg.model_type == "deit":
        @torch.no_grad()
        def extract(images):
            features = model.forward_features(images)[:, 0]   # CLS token
            logits = model.head(features)
            return logits, features

    elif teacher_cfg.model_type == "resnet50":
        feat_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
        fc_head = model.fc

        @torch.no_grad()
        def extract(images):
            features = feat_extractor(images).flatten(1)
            logits = fc_head(features)
            return logits, features
    else:
        raise ValueError(f"Unknown vision-only type: {teacher_cfg.model_type}")

    # ── Extract ──
    all_logits = []
    all_features = []
    for images, _ in tqdm(loader, desc=f"Extracting {teacher_key} (fine-tuned)"):
        images = images.to(device)
        logits, features = extract(images)
        all_logits.append(logits.cpu())
        all_features.append(features.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_features = torch.cat(all_features, dim=0)

    # ── Dimension sanity check ──
    assert all_logits.shape[1] == ds_cfg.num_classes, (
        f"CRITICAL: logit dim {all_logits.shape[1]} != {ds_cfg.num_classes}. "
        f"Fine-tuned checkpoint may be corrupted."
    )

    torch.save(all_logits, logit_path)
    torch.save(all_features, feat_path)
    print(f"  Saved logits:   {all_logits.shape} → {logit_path}")
    print(f"  Saved features: {all_features.shape} → {feat_path}")

    if split == "val" or (split == "test" and dataset_name != "imagenet"):
        compute_and_save_accuracy(all_logits, loader, save_dir, teacher_key, split)


# ──────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────

def cache_teacher_outputs(
    teacher_key: str,
    dataset_name: str,
    split: str,
    data_root: str,
    cache_dir: str,
    device: str = "cuda",
    batch_size: int = 128,
    finetuned_dir: str = "./finetuned_teachers",
):
    """Cache logits and features for a single teacher on a dataset split.

    Automatically routes to VLM (zero-shot) or vision-only (fine-tuned) path.
    """
    teacher_cfg = TEACHERS[teacher_key]

    if teacher_cfg.requires_finetune:
        cache_vision_teacher(
            teacher_key, teacher_cfg, dataset_name, split,
            data_root, cache_dir, device, batch_size, finetuned_dir,
        )
    else:
        cache_vlm_teacher(
            teacher_key, teacher_cfg, dataset_name, split,
            data_root, cache_dir, device, batch_size,
        )


# ──────────────────────────────────────────────────────────
# Teacher Agreement Matrix
# ──────────────────────────────────────────────────────────

def compute_teacher_agreement(cache_dir: str, dataset_name: str, split: str,
                                teacher_keys: list):
    """Compute pairwise top-1 prediction agreement between teachers."""
    print(f"\n{'='*60}")
    print(f"  Teacher Agreement Matrix: {dataset_name}/{split}")
    print(f"{'='*60}")

    save_dir = os.path.join(cache_dir, dataset_name, split)
    logits = {}
    for key in teacher_keys:
        path = os.path.join(save_dir, f"{key}_logits.pt")
        if os.path.exists(path):
            logits[key] = torch.load(path, map_location="cpu", weights_only=True)

    if len(logits) < 2:
        print("  Need at least 2 cached teachers for agreement matrix.")
        return

    # ── Print per-teacher accuracy ──
    print("\n  Per-teacher accuracy:")
    for key in logits:
        score_path = os.path.join(save_dir, f"{key}_score.txt")
        if os.path.exists(score_path):
            with open(score_path) as f:
                acc = float(f.read().strip())
            tag = "VLM, zero-shot" if TEACHERS[key].is_vlm else "vision-only, fine-tuned"
            print(f"    {key:>24s}: {acc:6.2f}%  ({tag})")

    # ── Matrix ──
    keys = list(logits.keys())
    n = len(keys)
    print(f"\n  {'':>24s}", end="")
    for k in keys:
        print(f"  {k:>16s}", end="")
    print()

    for i in range(n):
        preds_i = logits[keys[i]].argmax(dim=1)
        print(f"  {keys[i]:>24s}", end="")
        for j in range(n):
            preds_j = logits[keys[j]].argmax(dim=1)
            agreement = (preds_i == preds_j).float().mean().item() * 100
            print(f"  {agreement:>15.1f}%", end="")
        print()

    print()
    for i in range(n):
        for j in range(i + 1, n):
            preds_i = logits[keys[i]].argmax(dim=1)
            preds_j = logits[keys[j]].argmax(dim=1)
            agreement = (preds_i == preds_j).float().mean().item() * 100
            if agreement > 90:
                print(f"  ⚠ WARNING: {keys[i]} and {keys[j]} agree {agreement:.1f}% — "
                      f"consider swapping one teacher for diversity.")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cache teacher outputs for VLMSlim",
        epilog="""
Order of operations:
  1. Fine-tune vision-only baselines first (if needed for Exp 0):
     python finetune_teacher.py --teacher all --dataset cifar100

  2. Then cache everything:
     python cache_teachers.py --dataset cifar100

  3. Or cache only VLM teachers (no fine-tuning needed):
     python cache_teachers.py --dataset cifar100 \\
         --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar100", "cub200", "imagenet"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--finetuned_dir", type=str, default="./finetuned_teachers")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--teachers", type=str, nargs="+",
                        default=["openclip_vitl14", "metaclip_vitb16", "clip_vitb16",
                                 "deit_vitb16", "resnet50_supervised"],
                        help="Teacher keys to cache")
    args = parser.parse_args()

    DATASETS[args.dataset].data_root = args.data_root

    # Determine splits
    if args.dataset == "cifar100":
        splits = ["train", "val", "test"]
    elif args.dataset == "cub200":
        splits = ["train", "test"]
    elif args.dataset == "imagenet":
        splits = ["train", "val"]

    # ── Warn about missing fine-tuned checkpoints ──
    vision_teachers = [t for t in args.teachers
                       if t in TEACHERS and TEACHERS[t].requires_finetune]
    if vision_teachers:
        from finetune_teacher import is_finetuned
        missing = [t for t in vision_teachers
                   if not is_finetuned(t, args.dataset, args.finetuned_dir)]
        if missing:
            print(f"\n  ⚠  Vision-only teachers need fine-tuning before caching:")
            for t in missing:
                print(f"     python finetune_teacher.py "
                      f"--teacher {t} --dataset {args.dataset}")
            print(f"  These will be skipped. VLM teachers will still be cached.\n")

    # ── Cache each teacher ──
    for teacher_key in args.teachers:
        if teacher_key not in TEACHERS:
            print(f"  [SKIP] Unknown teacher: {teacher_key}")
            continue
        for split in splits:
            cache_teacher_outputs(
                teacher_key=teacher_key,
                dataset_name=args.dataset,
                split=split,
                data_root=args.data_root,
                cache_dir=args.cache_dir,
                device=args.device,
                batch_size=args.batch_size,
                finetuned_dir=args.finetuned_dir,
            )

    # ── Agreement matrix over all successfully cached teachers ──
    eval_split = "val" if args.dataset in ("cifar100", "imagenet") else "test"
    cached_teachers = []
    for t in args.teachers:
        logit_path = os.path.join(
            args.cache_dir, args.dataset, eval_split, f"{t}_logits.pt"
        )
        if os.path.exists(logit_path):
            cached_teachers.append(t)

    if len(cached_teachers) >= 2:
        compute_teacher_agreement(
            args.cache_dir, args.dataset, eval_split, cached_teachers
        )


if __name__ == "__main__":
    main()
