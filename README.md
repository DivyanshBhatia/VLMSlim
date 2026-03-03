# VLMSlim — Experiment Codebase

**Stability-Aware Multi-Teacher Distillation from Vision-Language Models**

Target: BMVC 2026

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fine-tune vision-only baselines (needed for Exp 0 sanity check)
python finetune_teacher.py --teacher all --dataset cifar100

# 3. Cache all teacher outputs (run ONCE per dataset)
python cache_teachers.py --dataset cifar100 --data_root ./data --cache_dir ./cache

# 4. Run the sanity check first
python run_experiments.py --exp exp0 --seeds 42 123 456

# 5. If gate passes, run everything in sequence
python run_experiments.py --exp all --seeds 42 123 456

# 6. Generate all paper figures
python analyze.py --output_dir ./outputs --figures_dir ./figures
```

---

## Teacher Setup

### VLM Teachers (zero-shot — no fine-tuning needed)

| Teacher | Model | Zero-shot CIFAR-100 | Feature Dim |
|---------|-------|---------------------|-------------|
| `openclip_vitl14` | ViT-L/14 (LAION-2B) | 73.18% | 768 |
| `metaclip_vitb16` | ViT-B/16 (MetaCLIP 400M) | TBD | 512 |
| `clip_vitb16` | ViT-B/16 (OpenAI) | 62.20% | 512 |

MetaCLIP replaces SigLIP as the default third teacher. SigLIP scored only 21% zero-shot on CIFAR-100 — barely above chance on 100 classes. Its low agreement with CLIP/OpenCLIP (~20%) was noise, not diversity. SigLIP remains available in the codebase for datasets where it performs better.

### Vision-Only Baselines (must fine-tune before use)

| Teacher | Pretrained On | Requires |
|---------|--------------|----------|
| `deit_vitb16` | ImageNet-1K (1000 classes) | `finetune_teacher.py` |
| `resnet50_supervised` | ImageNet-1K (1000 classes) | `finetune_teacher.py` |

These models ship with 1000-class ImageNet heads. Using them directly on CIFAR-100 (100 classes) produces meaningless logits — the cached "accuracy" will be ~1% because the argmax lands on random ImageNet class indices. The `finetune_teacher.py` script replaces the head and trains to convergence, producing a fair baseline for the Exp 0 comparison.

```bash
# Fine-tune both vision-only teachers on CIFAR-100
python finetune_teacher.py --teacher all --dataset cifar100

# Fine-tune for CUB-200 (if needed)
python finetune_teacher.py --teacher all --dataset cub200
```

---

## Full Pipeline (Recommended Order)

### Week 0: Teacher Preparation

```bash
# Fine-tune vision-only baselines (~50 epochs each, ~2 hours total)
python finetune_teacher.py --teacher deit_vitb16 --dataset cifar100
python finetune_teacher.py --teacher resnet50_supervised --dataset cifar100

# Cache all teacher outputs
python cache_teachers.py --dataset cifar100 \
    --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16 deit_vitb16 resnet50_supervised
```

### Week 1: Sanity Check + Baseline

```bash
# Exp 0 — Does VLM teacher beat fine-tuned vision-only? (~6h)
python run_experiments.py --exp exp0

# Exp 1 — Concurrent weighted-average baseline (~3h × 3 seeds)
python run_experiments.py --exp exp1
```

### Week 2: Core Experiments

```bash
# Exp 2 — Naive sequential, proving the moving-goal problem (~3h × 3 seeds)
python run_experiments.py --exp exp2

# Exp 3 — Full VLMSlim (~6h × 3 seeds)
python run_experiments.py --exp exp3
```

### Week 3–4: Ablations + Sensitivity

```bash
# Exp 4 — All 8 ablation variants (~24h total)
python run_experiments.py --exp exp4

# λ sweep (~15h)
python run_experiments.py --exp lambda_sweep

# Phase boundary sensitivity (~9h)
python run_experiments.py --exp phase_sensitivity

# Teacher ordering (~9h)
python run_experiments.py --exp teacher_ordering
```

### Week 5–6: CUB-200

```bash
# Prepare CUB-200 teachers
python finetune_teacher.py --teacher all --dataset cub200
python cache_teachers.py --dataset cub200

# Run CUB-200 experiments
python run_experiments.py --exp exp5
```

### Background: ImageNet (Start Week 1)

```bash
# Cache ImageNet teachers (one-time, ~50 GB disk, ~3 hours)
# Note: vision-only teachers don't need fine-tuning on ImageNet
# since they're already pretrained on ImageNet-1K
python cache_teachers.py --dataset imagenet \
    --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16 --batch_size 256

# Run key ImageNet configs
python run_experiments.py --exp exp6
```

### Final: Generate All Figures

```bash
python analyze.py --output_dir ./outputs --figures_dir ./figures
```

---

## File Structure

```
vlmslim/
├── config.py              # Teachers, students, datasets, experiment presets
├── models.py              # Student/teacher model loading, projection adaptor Φ
├── datasets.py            # CIFAR-100, CUB-200, ImageNet loaders + cached teacher wrapper
├── finetune_teacher.py    # Fine-tune vision-only baselines on target dataset (NEW)
├── cache_teachers.py      # Pre-compute & save teacher logits/features to disk
├── losses.py              # CE, KD, feature alignment, anchor loss, cumulative targets
├── utils.py               # Seeding, metrics, logger, LR scheduler, teacher scoring
├── train.py               # Core training loop with phase scheduling
├── run_experiments.py     # Main orchestrator with go/no-go gates
├── analyze.py             # Generate all paper figures from experiment logs
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Hyperparameter Summary (v2 Simplified)

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| α (CE weight) | 0.1 | FIXED | Hinton et al. standard |
| β (KD weight) | 0.9 | FIXED | β = 1 − α |
| τ (temperature) | 4.0 | FIXED | Standard for KD |
| γ (feature weight) | auto | DERIVED | Set so L\_feat ≈ L\_KD at epoch 1 |
| **λ (anchor strength)** | **0.1** | **TUNED** | **Only tuned hyperparameter** |
| Phase boundaries | equal split | FIXED | total\_epochs / K per teacher |

### Note on α/β and Weak Teachers

The α=0.1, β=0.9 split assumes the teacher is strong (90%+ accuracy), which is standard in KD literature (Hinton et al., DKD, etc.). However, VLM teachers on CIFAR-100 are much weaker than typical KD teachers: OpenCLIP gets 73% zero-shot, CLIP gets 62%. Early Exp 0 results show that a student distilled from CLIP at α=0.1 reaches only ~71.4%, which underperforms a ResNet-18 trained from scratch (~77%). This is because β=0.9 forces the student to spend 90% of its gradient budget matching a 62%-accurate teacher.

If this pattern holds with OpenCLIP (73%), two fixes are available:

1. **Increase α** (e.g. α=0.5, β=0.5) to give the ground-truth label equal weight. This introduces a second hyperparameter but may be necessary for weak-teacher KD.
2. **Lean on features, not logits.** Increase the relative importance of the feature alignment path (γ) and reduce β. The VLM's value may lie in its feature geometry (cross-modal structure from language supervision), not its noisy top-1 predictions.

Both can be tested with a single-seed quick run before committing to full experiments.

---

## Output Structure

```
finetuned_teachers/                          # Fine-tuned vision-only checkpoints
├── deit_vitb16_cifar100/best.pth
└── resnet50_supervised_cifar100/best.pth

cache/                                       # Pre-computed teacher outputs
└── cifar100/
    ├── train/
    │   ├── openclip_vitl14_logits.pt        # (45000, 100)
    │   ├── openclip_vitl14_features.pt      # (45000, 768)
    │   ├── metaclip_vitb16_logits.pt
    │   └── ...
    ├── val/
    │   ├── openclip_vitl14_logits.pt        # (5000, 100)
    │   ├── openclip_vitl14_score.txt        # "73.18"
    │   └── ...
    └── test/
        └── ...

outputs/                                     # Experiment results
├── exp0_clip_vitb16/
│   └── cifar100_resnet18_seed42/
│       ├── experiment_log.json              # All 12 checklist metrics per epoch
│       └── best_model.pth
├── exp1_concurrent/
├── exp3_vlmslim_lam0.1/
├── exp4_h_full/
└── lambda_sweep_lam0.05/

figures/                                     # Paper figures
├── fig2_hero_stability_curves.pdf
├── fig3_ablation_bars.pdf
├── fig4_lambda_sensitivity.pdf
├── fig5_feature_distance.pdf
└── supp_gradient_norms.pdf
```

---

## Key Design Decisions

1. **Offline teacher caching.** Teacher logits and features are pre-computed once and saved to disk. This avoids running 3 VLM forward passes during student training, reducing GPU-hours by ~3×.

2. **Automatic γ derivation.** The feature alignment weight is computed at epoch 1 by measuring the ratio of KD loss to feature loss over 100 batches. This eliminates a hyperparameter and makes the method reproducible.

3. **Fine-tuned vision-only baselines.** DeiT and ResNet-50 are fine-tuned on the target dataset before use as KD teachers. Without this step, their 1000-class ImageNet heads produce meaningless logits on CIFAR-100 or CUB-200, invalidating the Exp 0 comparison.

4. **MetaCLIP over SigLIP.** SigLIP's sigmoid-based training objective yields only 21% zero-shot accuracy on CIFAR-100 — too weak to contribute useful dark knowledge. MetaCLIP (ViT-B/16, trained on 400M curated pairs) uses the same contrastive objective as CLIP/OpenCLIP, ensuring all three teachers share a compatible representation space while differing in training data.

5. **Go/no-go gates.** The runner checks critical assumptions at each stage and stops early if a kill switch triggers. Use `--skip_gates` to override.

6. **Equal phase splitting.** Instead of tuning phase boundaries, training is divided equally among K teachers. Sensitivity experiments verify this is robust (≤1.5% variation).

---

## Troubleshooting

**"Cannot cache deit\_vitb16 / resnet50\_supervised"**: These are vision-only teachers that need fine-tuning first. Run `python finetune_teacher.py --teacher all --dataset cifar100` before caching.

**Logit dimension mismatch assertion**: A cached teacher's logits don't match the dataset's class count. This usually means an old cache from before the fine-tuning fix. Delete the relevant files in `./cache/` and re-run caching.

**VLM student underperforms scratch-trained baseline**: Check the α/β ratio. With weak VLM teachers (< 75% accuracy), α=0.1 gives too little weight to ground-truth labels. Try α=0.3 or α=0.5.

**Out of VRAM**: Reduce `batch_size` in the dataset config, or use gradient accumulation. Teacher caching is the most memory-hungry step (loads full VLM models).

**Teacher caching is slow**: Use `--batch_size 256` on A100. For ImageNet, expect ~3 hours per teacher.

**Missing CUB-200 data**: Download from https://data.caltech.edu/records/65de6-vp158 and extract to `./data/CUB_200_2011/`.

**CIFAR-100 resolution warning**: Expected. CIFAR images are 32×32 but VLM teachers need 224×224. The caching script upscales automatically. Students train at native 32×32.
