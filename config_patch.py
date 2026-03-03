"""
VLMSlim — config.py PATCH INSTRUCTIONS
========================================
Apply these changes to your existing config.py.

Three changes:
  1. Add `requires_finetune` field to TeacherConfig
  2. Replace SigLIP with MetaCLIP as default third teacher
  3. Add DEFAULT_VLM_TEACHERS constant
  4. Replace "siglip_vitb16" with "metaclip_vitb16" in all experiment presets
"""

# ──────────────────────────────────────────────────────────
# CHANGE 1: Update TeacherConfig dataclass
# ──────────────────────────────────────────────────────────
#
# OLD:
#   @dataclass
#   class TeacherConfig:
#       name: str
#       model_type: str
#       checkpoint: str
#       feature_dim: int
#       is_vlm: bool = True
#
# NEW:
#   @dataclass
#   class TeacherConfig:
#       name: str
#       model_type: str
#       checkpoint: str
#       feature_dim: int
#       is_vlm: bool = True
#       requires_finetune: bool = False   # <── NEW FIELD


# ──────────────────────────────────────────────────────────
# CHANGE 2: Replace TEACHERS dict entirely
# ──────────────────────────────────────────────────────────

TEACHERS_NEW = {
    # ── VLM teachers (zero-shot capable, no fine-tuning) ──
    "clip_vitb16": {
        "name": "clip_vitb16",
        "model_type": "clip",
        "checkpoint": "ViT-B-16/openai",
        "feature_dim": 512,
        "is_vlm": True,
        "requires_finetune": False,
    },
    "openclip_vitl14": {
        "name": "openclip_vitl14",
        "model_type": "openclip",
        "checkpoint": "ViT-L-14/laion2b_s32b_b82k",
        "feature_dim": 768,
        "is_vlm": True,
        "requires_finetune": False,
    },
    # MetaCLIP replaces SigLIP
    # (SigLIP scored 21% zero-shot on CIFAR-100 — too weak to be useful)
    "metaclip_vitb16": {
        "name": "metaclip_vitb16",
        "model_type": "openclip",                          # <── uses OpenCLIP loader
        "checkpoint": "ViT-B-16-quickgelu/metaclip_400m",  # <── MetaCLIP weights
        "feature_dim": 512,
        "is_vlm": True,
        "requires_finetune": False,
    },
    # Keep SigLIP available (may work better on other datasets)
    "siglip_vitb16": {
        "name": "siglip_vitb16",
        "model_type": "siglip",
        "checkpoint": "google/siglip-base-patch16-224",
        "feature_dim": 768,
        "is_vlm": True,
        "requires_finetune": False,
    },

    # ── Vision-only baselines (MUST fine-tune before caching) ──
    "deit_vitb16": {
        "name": "deit_vitb16",
        "model_type": "deit",
        "checkpoint": "facebook/deit-base-patch16-224",
        "feature_dim": 768,
        "is_vlm": False,
        "requires_finetune": True,    # <── NEW: prevents 1000-class bug
    },
    "resnet50_supervised": {
        "name": "resnet50_supervised",
        "model_type": "resnet50",
        "checkpoint": "torchvision",
        "feature_dim": 2048,
        "is_vlm": False,
        "requires_finetune": True,    # <── NEW: prevents 1000-class bug
    },
}


# ──────────────────────────────────────────────────────────
# CHANGE 3: Add this constant after the TEACHERS dict
# ──────────────────────────────────────────────────────────

DEFAULT_VLM_TEACHERS = ["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"]


# ──────────────────────────────────────────────────────────
# CHANGE 4: Global find-and-replace in experiment presets
# ──────────────────────────────────────────────────────────
#
# Replace every occurrence of:
#     "openclip_vitl14", "siglip_vitb16", "clip_vitb16"
# With:
#     "openclip_vitl14", "metaclip_vitb16", "clip_vitb16"
#
# This affects: exp1_concurrent, exp2_naive_sequential, exp3_vlmslim,
# exp4_ablation (the default teachers fallback), and the default_factory
# in ExperimentConfig.teachers.
#
# In sed:
#   sed -i 's/siglip_vitb16/metaclip_vitb16/g' config.py
