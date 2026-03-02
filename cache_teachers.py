"""
VLMSlim — Teacher Output Caching
==================================
Pre-compute and save teacher logits + features to disk.
This is run ONCE per dataset before any experiments.

Usage:
    python cache_teachers.py --dataset cifar100 --data_root ./data --cache_dir ./cache
    python cache_teachers.py --dataset cub200   --data_root ./data --cache_dir ./cache
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


def get_teacher_transform(teacher_cfg: TeacherConfig, teacher_size: int = 224):
    """Get the appropriate preprocessing transform for a teacher."""
    # Most VLMs use ImageNet normalization at 224×224
    return transforms.Compose([
        transforms.Resize((teacher_size, teacher_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_zero_shot_classifier(model, tokenizer, class_names, templates, device,
                                 model_type="openclip"):
    """Build a zero-shot classification weight matrix from text embeddings.

    Returns: (feature_dim, num_classes) weight matrix
    """
    import open_clip

    zeroshot_weights = []
    with torch.no_grad():
        for classname in tqdm(class_names, desc="Building zero-shot classifier"):
            texts = [t.format(classname) for t in templates]

            if model_type in ("clip", "openclip"):
                tokens = tokenizer(texts).to(device)
                text_features = model.encode_text(tokens)
            elif model_type == "siglip":
                # SigLIP uses its own processor
                inputs = tokenizer(text=texts, padding="max_length", truncation=True,
                                    return_tensors="pt").to(device)
                text_features = model.text_model(**{k: v for k, v in inputs.items()
                                                     if k != "pixel_values"}).pooler_output
            else:
                raise ValueError(f"Cannot build zero-shot classifier for {model_type}")

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0)
            text_features = text_features / text_features.norm()
            zeroshot_weights.append(text_features)

    return torch.stack(zeroshot_weights, dim=1)  # (feat_dim, num_classes)


def cache_teacher_outputs(
    teacher_key: str,
    dataset_name: str,
    split: str,
    data_root: str,
    cache_dir: str,
    device: str = "cuda",
    batch_size: int = 128,
):
    """Cache logits and features for a single teacher on a dataset split."""

    teacher_cfg = TEACHERS[teacher_key]
    ds_cfg = DATASETS[dataset_name]
    save_dir = os.path.join(cache_dir, dataset_name, split)
    os.makedirs(save_dir, exist_ok=True)

    logit_path = os.path.join(save_dir, f"{teacher_key}_logits.pt")
    feat_path = os.path.join(save_dir, f"{teacher_key}_features.pt")

    if os.path.exists(logit_path) and os.path.exists(feat_path):
        print(f"  [SKIP] {teacher_key}/{dataset_name}/{split} already cached.")
        return

    print(f"\n{'='*60}")
    print(f"  Caching: {teacher_key} on {dataset_name}/{split}")
    print(f"{'='*60}")

    # ── Load raw dataset (no augmentation, teacher-sized) ──
    teacher_transform = get_teacher_transform(teacher_cfg, teacher_size=224)

    if dataset_name == "cifar100":
        is_train = (split in ("train", "val"))
        raw_dataset = datasets.CIFAR100(data_root, train=is_train, download=True,
                                         transform=teacher_transform)
        if split == "train":
            raw_dataset = Subset(raw_dataset, list(range(45000)))
        elif split == "val":
            raw_dataset = Subset(raw_dataset, list(range(45000, 50000)))
        class_names = raw_dataset.dataset.classes if hasattr(raw_dataset, 'dataset') else raw_dataset.classes
        # CIFAR-100 class names
        if isinstance(raw_dataset, Subset):
            class_names = raw_dataset.dataset.classes
        else:
            class_names = raw_dataset.classes

    elif dataset_name == "cub200":
        is_train = (split in ("train", "val"))
        raw_dataset = CUB200Dataset(data_root, train=is_train, transform=teacher_transform)
        class_names = CUB200Dataset.get_class_names(data_root)

    elif dataset_name == "imagenet":
        folder = "train" if split == "train" else "val"
        raw_dataset = datasets.ImageFolder(
            os.path.join(data_root, "imagenet", folder),
            transform=teacher_transform
        )
        # ImageNet class names from folder structure
        class_names = [name.replace("_", " ") for name in raw_dataset.classes]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    # ── Load teacher model ──
    if teacher_cfg.model_type in ("clip", "openclip"):
        import open_clip
        model_name, pretrained = teacher_cfg.checkpoint.split("/")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        # Build zero-shot classifier
        templates = [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a photo of the large {}.",
            "a photo of the small {}.",
            "a photo of a {}, a type of object.",
        ]
        if dataset_name == "cub200":
            templates = ["a photo of a {}, a type of bird."]

        zs_weights = build_zero_shot_classifier(
            model, tokenizer, class_names, templates, device,
            model_type=teacher_cfg.model_type
        )  # (feat_dim, num_classes)

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

        templates = ["a photo of a {}."]
        if dataset_name == "cub200":
            templates = ["a photo of a {}, a type of bird."]

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

    elif teacher_cfg.model_type == "deit":
        import timm
        model = timm.create_model("deit_base_patch16_224", pretrained=True,
                                    num_classes=ds_cfg.num_classes).to(device).eval()

        @torch.no_grad()
        def extract(images):
            features = model.forward_features(images)[:, 0]  # CLS token
            logits = model.head(features)
            return logits, features

    elif teacher_cfg.model_type == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
        feat_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        @torch.no_grad()
        def extract(images):
            features = feat_extractor(images).flatten(1)
            logits = model.fc(features)
            return logits, features

    else:
        raise ValueError(f"Unknown teacher type: {teacher_cfg.model_type}")

    # ── Run extraction ──
    all_logits = []
    all_features = []

    for images, _ in tqdm(loader, desc=f"Extracting {teacher_key}"):
        images = images.to(device)
        logits, features = extract(images)
        all_logits.append(logits.cpu())
        all_features.append(features.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_features = torch.cat(all_features, dim=0)

    torch.save(all_logits, logit_path)
    torch.save(all_features, feat_path)
    print(f"  Saved logits:   {all_logits.shape} → {logit_path}")
    print(f"  Saved features: {all_features.shape} → {feat_path}")

    # ── Compute zero-shot accuracy ──
    if split == "val" or (split == "test" and dataset_name != "imagenet"):
        correct = 0
        total = 0
        for images, labels in loader:
            total += labels.size(0)
        preds = all_logits.argmax(dim=1)
        # We need actual labels; collect them
        all_labels = []
        for _, labels in loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels)
        correct = (preds == all_labels).sum().item()
        acc = 100.0 * correct / len(all_labels)
        print(f"  Zero-shot accuracy ({split}): {acc:.2f}%")

        # Save score for teacher ordering
        score_path = os.path.join(save_dir, f"{teacher_key}_score.txt")
        with open(score_path, "w") as f:
            f.write(f"{acc:.4f}")


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
        print("  Need at least 2 teachers for agreement matrix.")
        return

    keys = list(logits.keys())
    n = len(keys)
    print(f"\n  {'':>20s}", end="")
    for k in keys:
        print(f"  {k:>16s}", end="")
    print()

    for i in range(n):
        preds_i = logits[keys[i]].argmax(dim=1)
        print(f"  {keys[i]:>20s}", end="")
        for j in range(n):
            preds_j = logits[keys[j]].argmax(dim=1)
            agreement = (preds_i == preds_j).float().mean().item() * 100
            print(f"  {agreement:>15.1f}%", end="")
        print()

    print()
    # Check for >90% agreement warning
    for i in range(n):
        for j in range(i + 1, n):
            preds_i = logits[keys[i]].argmax(dim=1)
            preds_j = logits[keys[j]].argmax(dim=1)
            agreement = (preds_i == preds_j).float().mean().item() * 100
            if agreement > 90:
                print(f"  ⚠ WARNING: {keys[i]} and {keys[j]} agree {agreement:.1f}% — "
                      f"consider swapping one teacher for diversity.")


def main():
    parser = argparse.ArgumentParser(description="Cache teacher outputs for VLMSlim")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar100", "cub200", "imagenet"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--teachers", type=str, nargs="+",
                        default=["openclip_vitl14", "siglip_vitb16", "clip_vitb16",
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
            )

    # Compute agreement matrix on val/test
    vlm_teachers = [t for t in args.teachers if TEACHERS.get(t, None) and TEACHERS[t].is_vlm]
    eval_split = "val" if args.dataset in ("cifar100", "imagenet") else "test"
    if len(vlm_teachers) >= 2:
        compute_teacher_agreement(args.cache_dir, args.dataset, eval_split, vlm_teachers)


if __name__ == "__main__":
    main()
