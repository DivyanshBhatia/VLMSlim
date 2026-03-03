"""
VLMSlim Experiment Configurations
=================================
All hyperparameters follow the simplified v2 design:
  - 1 tuned parameter: λ (anchor strength)
  - 4 fixed/derived: α=0.1, β=0.9, τ=4.0, γ=auto
  - Phase boundaries: equal split (total_epochs / K)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


@dataclass
class TeacherConfig:
    name: str
    model_type: str          # "clip", "openclip", "deit", "resnet50"
    checkpoint: str          # HuggingFace or OpenCLIP checkpoint string
    feature_dim: int         # Output feature dimension
    is_vlm: bool = True      # False for vision-only baselines
    requires_finetune: bool = False  # True = must fine-tune on target dataset before caching


# ──────────────────────────────────────────────────────────
# Teacher definitions
# ──────────────────────────────────────────────────────────

TEACHERS = {
    # ── VLM teachers (zero-shot capable, no fine-tuning needed) ──
    "clip_vitb16": TeacherConfig(
        name="clip_vitb16", model_type="clip",
        checkpoint="ViT-B-16/openai",
        feature_dim=512, is_vlm=True,
    ),
    "openclip_vitl14": TeacherConfig(
        name="openclip_vitl14", model_type="openclip",
        checkpoint="ViT-L-14/laion2b_s32b_b82k",
        feature_dim=768, is_vlm=True,
    ),
    # MetaCLIP replaces SigLIP (SigLIP scored 21% zero-shot on CIFAR-100 — too weak)
    "metaclip_vitb16": TeacherConfig(
        name="metaclip_vitb16", model_type="openclip",
        checkpoint="ViT-B-16-quickgelu/metaclip_400m",
        feature_dim=512, is_vlm=True,
    ),
    # Keep SigLIP available but not in defaults — may work better on other datasets
    "siglip_vitb16": TeacherConfig(
        name="siglip_vitb16", model_type="siglip",
        checkpoint="google/siglip-base-patch16-224",
        feature_dim=768, is_vlm=True,
    ),

    # ── Vision-only baselines for Exp 0 ──
    # These MUST be fine-tuned on the target dataset before caching.
    # Run: python finetune_teacher.py --teacher deit_vitb16 --dataset cifar100
    "deit_vitb16": TeacherConfig(
        name="deit_vitb16", model_type="deit",
        checkpoint="facebook/deit-base-patch16-224",
        feature_dim=768, is_vlm=False,
        requires_finetune=True,
    ),
    "resnet50_supervised": TeacherConfig(
        name="resnet50_supervised", model_type="resnet50",
        checkpoint="torchvision",
        feature_dim=2048, is_vlm=False,
        requires_finetune=True,
    ),
}

# Default VLM teacher set (used in Exp 1–6)
DEFAULT_VLM_TEACHERS = ["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"]


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    train_size: Tuple[int, int]   # (H, W) for student
    teacher_size: Tuple[int, int] # (H, W) for teacher
    batch_size: int
    total_epochs: int
    lr: float
    data_root: str = "./data"


DATASETS = {
    "cifar100": DatasetConfig(
        name="cifar100", num_classes=100,
        train_size=(32, 32), teacher_size=(224, 224),
        batch_size=256, total_epochs=240, lr=0.1,
    ),
    "cub200": DatasetConfig(
        name="cub200", num_classes=200,
        train_size=(224, 224), teacher_size=(224, 224),
        batch_size=64, total_epochs=200, lr=0.01,
    ),
    "imagenet": DatasetConfig(
        name="imagenet", num_classes=1000,
        train_size=(224, 224), teacher_size=(224, 224),
        batch_size=512, total_epochs=100, lr=0.1,
    ),
}


@dataclass
class StudentConfig:
    name: str               # "resnet18", "mobilenetv2", "efficientnet_b0"
    feature_dim: int        # Penultimate feature dimension


STUDENTS = {
    "resnet18": StudentConfig(name="resnet18", feature_dim=512),
    "mobilenetv2": StudentConfig(name="mobilenetv2", feature_dim=1280),
    "efficientnet_b0": StudentConfig(name="efficientnet_b0", feature_dim=1280),
}


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # ── Experiment identity ──
    exp_name: str
    exp_id: str                              # e.g., "exp0_sanity", "exp3_vlmslim"
    seed: int = 42

    # ── Dataset ──
    dataset: str = "cifar100"

    # ── Student ──
    student: str = "resnet18"

    # ── Teachers (ordered best → worst by zero-shot score) ──
    teachers: List[str] = field(default_factory=lambda: [
        "openclip_vitl14", "metaclip_vitb16", "clip_vitb16"
    ])

    # ── Fixed hyperparameters (NOT tuned) ──
    alpha: float = 0.1          # CE weight — Hinton et al. standard
    beta: float = 0.9           # KD weight = 1 - alpha
    tau: float = 4.0            # Temperature — standard for KD
    # gamma is derived automatically, not set here

    # ── Tuned hyperparameter ──
    lam: float = 0.1            # λ — anchor strength (ONLY tuned param)

    # ── Training ──
    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    num_workers: int = 4

    # ── Method switches ──
    use_cumulative_targets: bool = True     # Cumulative ensemble logits
    use_anchor: bool = True                 # Anchor regularization
    use_feature_path: bool = True           # Feature alignment loss
    sequential: bool = True                 # Sequential teacher scheduling

    # ── Phase schedule ──
    # "equal" = total_epochs / K per teacher (default, no tuning)
    # Can also pass explicit list like [80, 160] for custom boundaries
    phase_schedule: str = "equal"

    # ── Paths ──
    cache_dir: str = "./cache"              # Pre-computed teacher outputs
    output_dir: str = "./outputs"
    log_dir: str = "./logs"

    # ── Logging ──
    use_wandb: bool = False
    wandb_project: str = "vlmslim"

    def get_dataset_config(self) -> DatasetConfig:
        return DATASETS[self.dataset]

    def get_student_config(self) -> StudentConfig:
        return STUDENTS[self.student]

    def get_teacher_configs(self) -> List[TeacherConfig]:
        return [TEACHERS[t] for t in self.teachers]

    def get_phase_boundaries(self) -> List[int]:
        """Return epoch numbers where teacher transitions happen."""
        ds = self.get_dataset_config()
        K = len(self.teachers)
        if self.phase_schedule == "equal":
            phase_len = ds.total_epochs // K
            return [phase_len * (i + 1) for i in range(K - 1)]
        else:
            # Custom boundaries passed as comma-separated string
            return [int(x) for x in self.phase_schedule.split(",")]

    def get_output_path(self) -> str:
        path = os.path.join(
            self.output_dir, self.exp_id,
            f"{self.dataset}_{self.student}_seed{self.seed}"
        )
        os.makedirs(path, exist_ok=True)
        return path


# ──────────────────────────────────────────────────────────
# Pre-built experiment configs
# ──────────────────────────────────────────────────────────

def exp0_sanity(teacher_key: str, seed: int = 42) -> ExperimentConfig:
    """Exp 0: Single teacher KD (VLM vs vision-only)."""
    return ExperimentConfig(
        exp_name=f"Sanity Check: {teacher_key}",
        exp_id=f"exp0_{teacher_key}",
        seed=seed,
        dataset="cifar100",
        student="resnet18",
        teachers=[teacher_key],
        use_cumulative_targets=False,
        use_anchor=False,
        use_feature_path=TEACHERS[teacher_key].is_vlm,  # Feature path only for VLM
        sequential=False,
        lam=0.0,
    )


def exp1_concurrent(seed: int = 42) -> ExperimentConfig:
    """Exp 1: Concurrent weighted-average of all 3 VLM teachers."""
    return ExperimentConfig(
        exp_name="Concurrent Weighted Average",
        exp_id="exp1_concurrent",
        seed=seed,
        dataset="cifar100",
        student="resnet18",
        teachers=["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"],
        use_cumulative_targets=False,  # Static average, not cumulative
        use_anchor=False,
        use_feature_path=True,
        sequential=False,              # All teachers from epoch 1
        lam=0.0,
    )


def exp2_naive_sequential(seed: int = 42) -> ExperimentConfig:
    """Exp 2: Naive sequential switching (no stability mechanisms)."""
    return ExperimentConfig(
        exp_name="Naive Sequential",
        exp_id="exp2_naive_seq",
        seed=seed,
        dataset="cifar100",
        student="resnet18",
        teachers=["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"],
        use_cumulative_targets=False,
        use_anchor=False,
        use_feature_path=True,
        sequential=True,
        lam=0.0,
    )


def exp3_vlmslim(seed: int = 42, lam: float = 0.1) -> ExperimentConfig:
    """Exp 3: Full VLMSlim method."""
    return ExperimentConfig(
        exp_name=f"VLMSlim (λ={lam})",
        exp_id=f"exp3_vlmslim_lam{lam}",
        seed=seed,
        dataset="cifar100",
        student="resnet18",
        teachers=["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"],
        use_cumulative_targets=True,
        use_anchor=True,
        use_feature_path=True,
        sequential=True,
        lam=lam,
    )


def exp4_ablation(variant: str, seed: int = 42) -> ExperimentConfig:
    """Exp 4: Ablation variants."""
    configs = {
        "a_vanilla_kd": dict(
            teachers=["resnet50_supervised"],
            use_cumulative_targets=False, use_anchor=False,
            use_feature_path=False, sequential=False, lam=0.0,
        ),
        "b_single_vlm": dict(
            teachers=["openclip_vitl14"],
            use_cumulative_targets=False, use_anchor=False,
            use_feature_path=True, sequential=False, lam=0.0,
        ),
        "c_concurrent": dict(
            use_cumulative_targets=False, use_anchor=False,
            use_feature_path=True, sequential=False, lam=0.0,
        ),
        "d_naive_seq": dict(
            use_cumulative_targets=False, use_anchor=False,
            use_feature_path=True, sequential=True, lam=0.0,
        ),
        "e_no_anchor": dict(
            use_cumulative_targets=True, use_anchor=False,
            use_feature_path=True, sequential=True, lam=0.0,
        ),
        "f_no_cumulative": dict(
            use_cumulative_targets=False, use_anchor=True,
            use_feature_path=True, sequential=True, lam=0.1,
        ),
        "g_no_feature": dict(
            use_cumulative_targets=True, use_anchor=True,
            use_feature_path=False, sequential=True, lam=0.1,
        ),
        "h_full": dict(
            use_cumulative_targets=True, use_anchor=True,
            use_feature_path=True, sequential=True, lam=0.1,
        ),
    }
    v = configs[variant]
    teachers = v.pop("teachers", ["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"])
    return ExperimentConfig(
        exp_name=f"Ablation: {variant}",
        exp_id=f"exp4_{variant}",
        seed=seed,
        dataset="cifar100",
        student="resnet18",
        teachers=teachers,
        **v,
    )
