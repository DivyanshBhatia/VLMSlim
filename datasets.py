"""
VLMSlim — Dataset Loading
==========================
CIFAR-100, CUB-200-2011, ImageNet-1K with proper augmentations.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
import numpy as np

from config import DatasetConfig


# ──────────────────────────────────────────────────────────
# Normalization constants
# ──────────────────────────────────────────────────────────

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────
# Cached teacher dataset wrapper
# ──────────────────────────────────────────────────────────

class CachedTeacherDataset(Dataset):
    """Wraps a base dataset and pairs each sample with pre-computed teacher outputs.

    Args:
        base_dataset:  The underlying image dataset
        cache_dir:     Path containing teacher cache files
        teacher_names: List of teacher keys in order
    """

    def __init__(self, base_dataset, cache_dir: str, teacher_names: list):
        self.base_dataset = base_dataset
        self.teacher_names = teacher_names

        # Load cached logits and features for each teacher
        self.teacher_logits = {}
        self.teacher_features = {}
        for name in teacher_names:
            logit_path = os.path.join(cache_dir, f"{name}_logits.pt")
            feat_path = os.path.join(cache_dir, f"{name}_features.pt")
            if os.path.exists(logit_path):
                self.teacher_logits[name] = torch.load(logit_path, map_location="cpu",
                                                        weights_only=True)
                print(f"  Loaded cached logits: {name} → {self.teacher_logits[name].shape}")
            if os.path.exists(feat_path):
                self.teacher_features[name] = torch.load(feat_path, map_location="cpu",
                                                          weights_only=True)
                print(f"  Loaded cached features: {name} → {self.teacher_features[name].shape}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        teacher_data = {}
        for name in self.teacher_names:
            td = {}
            if name in self.teacher_logits:
                td["logits"] = self.teacher_logits[name][idx]
            if name in self.teacher_features:
                td["features"] = self.teacher_features[name][idx]
            teacher_data[name] = td

        return img, label, teacher_data


# ──────────────────────────────────────────────────────────
# CIFAR-100
# ──────────────────────────────────────────────────────────

def get_cifar100(cfg: DatasetConfig, cache_dir: str = None, teacher_names: list = None):
    """Return CIFAR-100 train/val/test loaders.

    Train: 45,000 (with augmentation)
    Val:   5,000  (last 5k of original train, no augmentation)
    Test:  10,000 (no augmentation)
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = datasets.CIFAR100(cfg.data_root, train=True, download=True,
                                    transform=train_transform)
    full_train_eval = datasets.CIFAR100(cfg.data_root, train=True, download=True,
                                         transform=eval_transform)
    test_set = datasets.CIFAR100(cfg.data_root, train=False, download=True,
                                  transform=eval_transform)

    # Split: first 45k for training, last 5k for validation
    n_train = 45000
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, 50000))

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    # Wrap with cached teacher outputs if available
    if cache_dir and teacher_names:
        train_cache = os.path.join(cache_dir, "cifar100", "train")
        val_cache = os.path.join(cache_dir, "cifar100", "val")
        test_cache = os.path.join(cache_dir, "cifar100", "test")

        if os.path.exists(train_cache):
            print("Loading cached teacher outputs for CIFAR-100 train...")
            train_set = CachedTeacherDataset(train_set, train_cache, teacher_names)
        if os.path.exists(val_cache):
            print("Loading cached teacher outputs for CIFAR-100 val...")
            val_set = CachedTeacherDataset(val_set, val_cache, teacher_names)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') else 4,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────
# CUB-200-2011
# ──────────────────────────────────────────────────────────

class CUB200Dataset(Dataset):
    """CUB-200-2011 dataset loader.

    Expects the standard directory structure:
        data_root/CUB_200_2011/images/
        data_root/CUB_200_2011/image_class_labels.txt
        data_root/CUB_200_2011/train_test_split.txt
        data_root/CUB_200_2011/images.txt
        data_root/CUB_200_2011/classes.txt
    """

    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = os.path.join(root, "CUB_200_2011")
        self.transform = transform
        self.train = train

        # Parse metadata files
        images = self._read_lines("images.txt")
        labels = self._read_lines("image_class_labels.txt")
        split = self._read_lines("train_test_split.txt")

        self.samples = []
        for img_line, lbl_line, spl_line in zip(images, labels, split):
            img_id, img_path = img_line.split()
            _, label = lbl_line.split()
            _, is_train = spl_line.split()

            if (train and is_train == "1") or (not train and is_train == "0"):
                full_path = os.path.join(self.root, "images", img_path)
                self.samples.append((full_path, int(label) - 1))  # 0-indexed

    def _read_lines(self, filename):
        with open(os.path.join(self.root, filename), "r") as f:
            return f.read().strip().split("\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def get_class_names(root: str):
        """Return list of 200 species names for zero-shot prompts."""
        path = os.path.join(root, "CUB_200_2011", "classes.txt")
        names = []
        with open(path, "r") as f:
            for line in f:
                # Format: "001.Black_footed_Albatross"
                parts = line.strip().split(" ")[1]
                name = parts.split(".")[1].replace("_", " ")
                names.append(name)
        return names


def get_cub200(cfg: DatasetConfig, cache_dir: str = None, teacher_names: list = None):
    """Return CUB-200 train/val/test loaders.

    Train: 5,394 (90% of official train)
    Val:   600   (10% of official train)
    Test:  5,794 (official test)
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    full_train = CUB200Dataset(cfg.data_root, train=True, transform=train_transform)
    full_train_eval = CUB200Dataset(cfg.data_root, train=True, transform=eval_transform)
    test_set = CUB200Dataset(cfg.data_root, train=False, transform=eval_transform)

    # 90/10 split for train/val
    n_total = len(full_train)
    n_val = 600
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(42)  # Fixed split across experiments
    train_indices, val_indices = random_split(range(n_total), [n_train, n_val], generator=gen)

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    if cache_dir and teacher_names:
        train_cache = os.path.join(cache_dir, "cub200", "train")
        if os.path.exists(train_cache):
            print("Loading cached teacher outputs for CUB-200 train...")
            train_set = CachedTeacherDataset(train_set, train_cache, teacher_names)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size * 2, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────
# ImageNet-1K
# ──────────────────────────────────────────────────────────

def get_imagenet(cfg: DatasetConfig, cache_dir: str = None, teacher_names: list = None):
    """Return ImageNet-1K train/val loaders.

    Expects: data_root/imagenet/train/ and data_root/imagenet/val/
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
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

    train_root = os.path.join(cfg.data_root, "imagenet", "train")
    val_root = os.path.join(cfg.data_root, "imagenet", "val")

    train_set = datasets.ImageFolder(train_root, transform=train_transform)
    val_set = datasets.ImageFolder(val_root, transform=eval_transform)

    if cache_dir and teacher_names:
        train_cache = os.path.join(cache_dir, "imagenet", "train")
        if os.path.exists(train_cache):
            print("Loading cached teacher outputs for ImageNet train...")
            train_set = CachedTeacherDataset(train_set, train_cache, teacher_names)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    return train_loader, val_loader, val_loader  # No separate test set for ImageNet


# ──────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────

def get_dataloaders(dataset_name: str, cfg: DatasetConfig,
                    cache_dir: str = None, teacher_names: list = None):
    """Get train/val/test loaders for any supported dataset."""
    if dataset_name == "cifar100":
        return get_cifar100(cfg, cache_dir, teacher_names)
    elif dataset_name == "cub200":
        return get_cub200(cfg, cache_dir, teacher_names)
    elif dataset_name == "imagenet":
        return get_imagenet(cfg, cache_dir, teacher_names)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
