# VLMSlim — Experiment Codebase

**Feature-Centric Multi-Teacher Distillation from Frozen Vision-Language Models**

Target: BMVC 2026

---

## Core Idea

Frozen VLM teachers (CLIP, OpenCLIP, MetaCLIP) have weak zero-shot logits on domain-specific datasets (62–73% on CIFAR-100) but geometrically rich feature spaces shaped by language supervision. VLMSlim treats **feature alignment as the primary distillation channel** and demotes KD logits to a lightweight regularizer. The stability-aware sequential scheduling ensures each teacher's feature structure is absorbed without catastrophic forgetting from teacher transitions.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Cache VLM teacher outputs (run ONCE per dataset — ~1 hour)
python cache_teachers.py --dataset cifar100 --data_root ./data --cache_dir ./cache \
    --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16

# 3. Run α/β diagnostic (~2h, single seed, 50 epochs per config)
#    This finds the right CE/KD/feature balance BEFORE committing
python diagnostic_sweep.py --dataset cifar100 --teacher openclip_vitl14

# 4. Update α/β in config.py based on diagnostic results, then run Exp 0
python run_experiments.py --exp exp0 --seeds 42

# 5. If gate passes, run everything
python run_experiments.py --exp all --seeds 42 123 456

# 6. Generate all paper figures
python analyze.py --output_dir ./outputs --figures_dir ./figures
```

---

## Why Frozen Teachers?

All VLM teachers are kept **frozen throughout** — no fine-tuning, no linear probes, no task-specific adaptation. This is deliberate:

"We keep all VLM teachers frozen to isolate the benefit of their pretrained representations, avoiding confounds from task-specific fine-tuning."

If you fine-tune the teachers, you're no longer distilling VLM knowledge — you're distilling a supervised classifier that happens to be initialized from a VLM. The cross-modal awareness argument disappears, and the contribution collapses to "we used a bigger pretrained model as teacher."

This creates a tension: frozen teacher logits are noisy (62–73% on CIFAR-100), but their feature geometry is rich. The loss architecture resolves this by making features primary and logits secondary.

---

## Teacher Setup

### VLM Teachers (frozen, zero-shot)

| Teacher | Model | Zero-shot CIFAR-100 | Feature Dim |
|---------|-------|---------------------|-------------|
| `openclip_vitl14` | ViT-L/14 (LAION-2B) | 73.18% | 768 |
| `metaclip_vitb16` | ViT-B/16 (MetaCLIP 400M) | TBD | 512 |
| `clip_vitb16` | ViT-B/16 (OpenAI) | 62.20% | 512 |

MetaCLIP replaces SigLIP as the default third teacher. SigLIP scored only 21% zero-shot on CIFAR-100 — barely above chance on 100 classes. SigLIP remains available in the codebase (`siglip_vitb16`) for datasets where it performs better.

### Vision-Only Baselines (optional, for extended ablations)

| Teacher | Pretrained On | Requires |
|---------|--------------|----------|
| `deit_vitb16` | ImageNet-1K (1000 classes) | `finetune_teacher.py` |
| `resnet50_supervised` | ImageNet-1K (1000 classes) | `finetune_teacher.py` |

These ship with 1000-class ImageNet heads and produce meaningless logits on CIFAR-100 without fine-tuning. They are **not** needed for the core experiments (Exp 1–6). They're only used in ablation variant `a_vanilla_kd` to compare against conventional supervised KD. If you need them:

```bash
python finetune_teacher.py --teacher all --dataset cifar100
```

---

## Loss Architecture (v3 — Feature-Centric)

```
L = α·L_CE + β·L_KD + γ·L_feat + λ·L_anchor
```

| Parameter | Value | Status | Notes |
|-----------|-------|--------|-------|
| α (CE weight) | 0.3 | **PROVISIONAL** | Validate with `diagnostic_sweep.py` |
| β (KD weight) | 0.2 | **PROVISIONAL** | Validate with `diagnostic_sweep.py` |
| τ (temperature) | 4.0 | FIXED | Standard for KD |
| γ (feature weight) | auto | DERIVED | γ = feature\_weight × mean\_CE / mean\_feat at epoch 1 |
| feature\_weight | 0.5 | FIXED | Features capture ~50% of gradient budget |
| **λ (anchor strength)** | **0.1** | **TUNED** | **Only tuned hyperparameter** |
| Phase boundaries | equal split | FIXED | total\_epochs / K per teacher |

### ⚠ α/β Must Be Validated First

The α=0.3, β=0.2 defaults are our best reasoning but have not been validated empirically. Before running full experiments, use the diagnostic sweep to find the right balance for your teacher/dataset:

```bash
python diagnostic_sweep.py --dataset cifar100 --teacher openclip_vitl14
```

This runs 4 configurations at 50 epochs each (~2h total):

| Config | α | β | Tests |
|--------|-----|-----|-------|
| A) feat\_only | 1.0 | 0.0 | CE + features, no KD logits at all |
| B) ce\_heavy | 0.5 | 0.1 | CE-dominated, KD as mild regularizer |
| C) balanced | 0.3 | 0.2 | Current default (feature-centric) |
| D) kd\_medium | 0.3 | 0.5 | More KD — do 73% logits actually help? |

Interpreting results:
- If A wins: KD logits from frozen VLMs are noise on this dataset. Set β=0.0.
- If C or B wins: Feature-centric balance is correct. Keep defaults.
- If D wins: Teacher logits are more useful than expected. Increase β.

Update α/β in `config.py` based on the winner, then run full experiments.

### Design Rationale

Standard KD (Hinton et al.) uses α=0.1, β=0.9 — designed for strong teachers at 90%+ accuracy. Our frozen VLM teachers are 62–73% on CIFAR-100. Early experiments showed that at α=0.1, a student distilled from CLIP reached only 71.4%, underperforming a scratch-trained ResNet-18 (~77%). The student was spending 90% of its gradient budget matching a teacher that's wrong on 38% of samples.

The hypothesis: feature alignment should be the dominant distillation channel. The VLM's value lies in its feature geometry (inter-class relationships shaped by 400M image-text pairs), not in its noisy top-1 predictions. The diagnostic sweep tests this before committing GPU-hours.

---

## Experiment Design

### Exp 0 — Sanity Check: Scratch vs VLM Feature Transfer

This is the critical gate. It runs:

1. **Scratch baseline** (pure CE, no teacher) — establishes the accuracy ceiling
2. **VLM distillation** (OpenCLIP, CLIP) — tests whether frozen VLM features transfer

The gate checks **feature quality**, not just accuracy. Specifically, the VLM student's inter/intra-class distance ratio must exceed the scratch baseline by ≥10%. The VLM student may trail on raw accuracy (since its teacher logits are noisy) but should produce a better-structured feature space.

### Exp 1–3 — Core Method

- **Exp 1**: Concurrent weighted-average of all 3 VLMs (static baseline)
- **Exp 2**: Naive sequential switching (proves the moving-goal problem)
- **Exp 3**: Full VLMSlim (cumulative targets + anchor + features)

### Exp 4 — Ablation Table (8 variants a–h)

Tests each component in isolation: vanilla KD, single VLM, concurrent, naive sequential, no-anchor, no-cumulative, no-feature, and full.

### Exp 5–6 — Generalization

- **Exp 5**: CUB-200 + sensitivity sweeps (λ, phase boundaries, teacher ordering)
- **Exp 6**: ImageNet-1K (key configs only, single seed)

---

## Full Pipeline (Recommended Order)

### Week 0: Teacher Preparation + Diagnostic

```bash
# Cache VLM teacher outputs (no fine-tuning needed)
python cache_teachers.py --dataset cifar100 \
    --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16

# Run α/β diagnostic (~2h) — do this BEFORE full experiments
python diagnostic_sweep.py --dataset cifar100 --teacher openclip_vitl14

# Review results table, update α/β in config.py if needed

# Optional: fine-tune vision-only baselines for ablation variant (a)
python finetune_teacher.py --teacher all --dataset cifar100
python cache_teachers.py --dataset cifar100 \
    --teachers deit_vitb16 resnet50_supervised
```

### Week 1: Sanity Check + Baseline

```bash
# Exp 0 — Scratch vs VLM feature transfer (~3h: 1 scratch + 2 VLM × 3 seeds)
python run_experiments.py --exp exp0

# Exp 1 — Concurrent weighted-average baseline (~3h × 3 seeds)
python run_experiments.py --exp exp1
```

### Week 2: Core Experiments

```bash
# Exp 2 — Naive sequential (~3h × 3 seeds)
python run_experiments.py --exp exp2

# Exp 3 — Full VLMSlim (~6h × 3 seeds)
python run_experiments.py --exp exp3
```

### Week 3–4: Ablations + Sensitivity

```bash
python run_experiments.py --exp exp4
python run_experiments.py --exp lambda_sweep
python run_experiments.py --exp phase_sensitivity
python run_experiments.py --exp teacher_ordering
```

### Week 5–6: CUB-200

```bash
python cache_teachers.py --dataset cub200 \
    --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16
python run_experiments.py --exp exp5
```

### Background: ImageNet (Start Week 1)

```bash
python cache_teachers.py --dataset imagenet \
    --teachers openclip_vitl14 metaclip_vitb16 clip_vitb16 --batch_size 256
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
├── finetune_teacher.py    # Fine-tune vision-only baselines on target dataset
├── cache_teachers.py      # Pre-compute & save teacher logits/features to disk
├── diagnostic_sweep.py    # Quick α/β sweep to validate loss weights before full runs
├── losses.py              # CE, KD, feature alignment, anchor loss, cumulative targets
├── utils.py               # Seeding, metrics, logger, LR scheduler, teacher scoring
├── train.py               # Core training loop with phase scheduling
├── run_experiments.py     # Main orchestrator with go/no-go gates
├── analyze.py             # Generate all paper figures from experiment logs
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Output Structure

```
cache/                                       # Pre-computed teacher outputs
└── cifar100/
    ├── train/
    │   ├── openclip_vitl14_logits.pt        # (45000, 100)
    │   ├── openclip_vitl14_features.pt      # (45000, 768)
    │   └── ...
    ├── val/
    │   ├── openclip_vitl14_score.txt        # "73.18"
    │   └── ...
    └── test/

finetuned_teachers/                          # Only if vision-only baselines used
├── deit_vitb16_cifar100/best.pth
└── resnet50_supervised_cifar100/best.pth

outputs/
├── diagnostic_openclip_vitl14_cifar100_summary.json   # α/β sweep results
├── diag_A_feat_only_openclip_vitl14/                  # 50-epoch diagnostic runs
├── diag_B_ce_heavy_openclip_vitl14/
├── diag_C_balanced_openclip_vitl14/
├── diag_D_kd_medium_openclip_vitl14/
├── exp0_scratch/                            # Scratch baseline (pure CE)
│   └── cifar100_resnet18_seed42/
│       ├── experiment_log.json              # Includes feature_metrics
│       └── best_model.pth
├── exp0_openclip_vitl14/                    # VLM distillation
├── exp1_concurrent/
├── exp3_vlmslim_lam0.1/
├── exp4_h_full/
└── lambda_sweep_lam0.05/

figures/
├── fig2_hero_stability_curves.pdf
├── fig3_ablation_bars.pdf
├── fig4_lambda_sensitivity.pdf
├── fig5_feature_distance.pdf               # KEY: shows feature ratio improvement
└── supp_gradient_norms.pdf
```

---

## Key Design Decisions

1. **Frozen teachers throughout.** VLM teachers are never fine-tuned. This isolates the benefit of language-supervised representations and cleanly differentiates VLMSlim from methods that adapt the teacher to the target task. Frozen teacher logits are noisy — that's expected and handled by the loss architecture.

2. **Feature-centric loss.** Features are the primary distillation channel (α=0.3, β=0.2, γ auto-scaled to ~50% budget). The VLM's value lies in its feature geometry, not its zero-shot classification accuracy. KD logits serve as a soft regularizer, not the main learning signal.

3. **α/β diagnostic before committing.** The α/β values are provisional. `diagnostic_sweep.py` runs 4 configurations at 50 epochs (~2h) to empirically validate whether KD logits help, hurt, or don't matter for a given teacher/dataset combination. This avoids wasting GPU-hours on a loss balance that might be wrong.

4. **Automatic γ derivation.** The feature alignment weight is computed at epoch 1 by measuring γ = feature_weight × mean_CE / mean_feat over 100 batches. This eliminates a hyperparameter and adapts to different teacher/dataset combinations automatically.

5. **Scratch baseline in Exp 0.** Instead of comparing VLM vs vision-only teachers, Exp 0 compares VLM-distilled students against a scratch-trained baseline on both accuracy and feature quality (inter/intra-class distance ratio). The gate checks feature structure improvement, not accuracy superiority.

6. **MetaCLIP over SigLIP.** SigLIP's sigmoid-based training yields only 21% zero-shot on CIFAR-100. MetaCLIP uses the same contrastive objective as CLIP/OpenCLIP, ensuring compatible representation spaces while differing in training data scale and curation.

7. **Offline teacher caching.** Teacher logits and features are pre-computed once and saved to disk, avoiding 3 VLM forward passes per student training batch.

8. **Go/no-go gates.** The runner checks critical assumptions at each stage. Exp 0 gates on feature quality, Exp 2 gates on the moving-goal problem, Exp 3 gates on VLMSlim beating the concurrent baseline. Use `--skip_gates` to override.

---

## Troubleshooting

**Exp 0 gate fails (feature ratio not improved):** Check that `collect_features=True` is working in the final evaluation. If feature metrics are missing from the JSON log, the issue is in `evaluate()` not collecting features at test time.

**VLM student accuracy trails scratch baseline:** This is expected with frozen teachers on domain-shifted datasets. The paper's claim is about feature quality, not raw accuracy. Check the inter/intra distance ratio — if it's 10%+ higher than scratch, the method is working as designed.

**"Cannot cache deit\_vitb16 / resnet50\_supervised":** These vision-only teachers need fine-tuning first. Run `python finetune_teacher.py --teacher all --dataset cifar100`. These are only needed for ablation variant `a_vanilla_kd`, not for the core experiments.

**Logit dimension mismatch assertion:** A cached teacher's logits don't match the dataset's class count. Delete the relevant files in `./cache/` and re-run caching.

**Out of VRAM:** Reduce `batch_size` in the dataset config. Teacher caching is the most memory-hungry step (loads full VLM models).

**Teacher caching is slow:** Use `--batch_size 256` on A100. For ImageNet, expect ~3 hours per teacher.

**Missing CUB-200 data:** Download from https://data.caltech.edu/records/65de6-vp158 and extract to `./data/CUB_200_2011/`.

**CIFAR-100 resolution warning:** Expected. CIFAR images are 32×32 but VLM teachers need 224×224. The caching script upscales automatically. Students train at native 32×32.

---

## Paper Framing (for reference)

The paper should state explicitly: "We keep all VLM teachers frozen to isolate the benefit of their pretrained representations." This is reviewer-proof and differentiates from methods like VL2Lite that benefit from task-adapted language branches.

The contribution is not "VLM teachers produce higher-accuracy students" (they may not, on domain-shifted datasets). The contribution is: "VLM feature geometry, shaped by cross-modal pretraining, transfers through distillation to produce students with richer representational structure, and stability-aware sequential scheduling makes multi-teacher feature transfer practical."
