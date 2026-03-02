# VLMSlim — Experiment Codebase

**Stability-Aware Multi-Teacher Distillation from Vision-Language Models**

Target: BMVC 2026

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Cache teacher outputs (run ONCE per dataset — takes ~1 hour for CIFAR-100)
python cache_teachers.py --dataset cifar100 --data_root ./data --cache_dir ./cache

# 3. Run the sanity check first (Exp 0 — ~6 hours)
python run_experiments.py --exp exp0 --seeds 42 123 456

# 4. If gate passes, run everything in sequence
python run_experiments.py --exp all --seeds 42 123 456

# 5. Generate all paper figures
python analyze.py --output_dir ./outputs --figures_dir ./figures
```

---

## Full Pipeline (Recommended Order)

### Week 1: Sanity + Baseline

```bash
# Step 1: Cache teacher outputs for CIFAR-100
python cache_teachers.py --dataset cifar100 \
    --teachers openclip_vitl14 siglip_vitb16 clip_vitb16 deit_vitb16 resnet50_supervised

# Step 2: Exp 0 — Does VLM teacher beat vision-only? (~6h)
python run_experiments.py --exp exp0

# Step 3: Exp 1 — Concurrent weighted-average baseline (~3h x 3 seeds)
python run_experiments.py --exp exp1
```

### Week 2: Core Experiments

```bash
# Step 4: Exp 2 — Naive sequential, proving the moving-goal problem (~3h x 3 seeds)
python run_experiments.py --exp exp2

# Step 5: Exp 3 — Full VLMSlim (~6h x 3 seeds)
python run_experiments.py --exp exp3
```

### Week 3-4: Ablations

```bash
# Step 6: Exp 4 — All 8 ablation variants (~24h total)
python run_experiments.py --exp exp4

# Step 7: Lambda sweep (~15h)
python run_experiments.py --exp lambda_sweep

# Step 8: Phase boundary sensitivity (~9h)
python run_experiments.py --exp phase_sensitivity

# Step 9: Teacher ordering (~9h)
python run_experiments.py --exp teacher_ordering
```

### Week 5-6: CUB-200

```bash
# Cache teachers for CUB-200
python cache_teachers.py --dataset cub200 --data_root ./data --cache_dir ./cache

# Run CUB-200 experiments
python run_experiments.py --exp exp5
```

### Background: ImageNet (Start Week 1)

```bash
# Cache ImageNet teachers (one-time, ~50 GB disk, ~3 hours)
python cache_teachers.py --dataset imagenet --data_root ./data --cache_dir ./cache \
    --teachers openclip_vitl14 siglip_vitb16 clip_vitb16 --batch_size 256

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
├── config.py            # All configs: teachers, students, datasets, experiment presets
├── models.py            # Student/teacher model loading, projection adaptor
├── datasets.py          # CIFAR-100, CUB-200, ImageNet loaders + cached teacher wrapper
├── cache_teachers.py    # Pre-compute and save teacher logits/features (run once)
├── losses.py            # CE, KD, feature alignment, anchor loss, cumulative targets
├── utils.py             # Seeding, metrics, logger, LR scheduler, teacher scoring
├── train.py             # Core training loop with phase scheduling
├── run_experiments.py   # Main orchestrator with go/no-go gates
├── analyze.py           # Generate all paper figures from experiment logs
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Hyperparameter Summary (v2 Simplified)

| Parameter | Value | Status |
|-----------|-------|--------|
| alpha (CE weight) | 0.1 | FIXED — Hinton et al. standard |
| beta (KD weight) | 0.9 | FIXED — beta = 1 - alpha |
| tau (temperature) | 4.0 | FIXED — standard for KD |
| gamma (feature weight) | auto | DERIVED — set so L_feat ~ L_KD at epoch 1 |
| **lambda (anchor strength)** | **0.1** | **TUNED — only tuned hyperparameter** |
| Phase boundaries | equal split | FIXED — total_epochs / K per teacher |

---

## Output Structure

After running experiments, the output directory looks like:

```
outputs/
├── exp0_clip_vitb16/
│   └── cifar100_resnet18_seed42/
│       ├── experiment_log.json     # Complete metrics (all 12 checklist items)
│       └── best_model.pth          # Best checkpoint
├── exp1_concurrent/
├── exp3_vlmslim_lam0.1/
├── exp4_h_full/
└── lambda_sweep_lam0.05/

figures/
├── fig2_hero_stability_curves.pdf   # HERO FIGURE
├── fig3_ablation_bars.pdf
├── fig4_lambda_sensitivity.pdf
├── fig5_feature_distance.pdf
└── supp_gradient_norms.pdf
```

---

## Troubleshooting

**Out of VRAM**: Reduce batch_size in the dataset config, or use gradient accumulation.
The caching step is the most memory-hungry (loading full VLM models).

**Teacher caching is slow**: Use --batch_size 256 for GPU-rich setups. For ImageNet,
this takes ~3 hours per teacher on a single A100.

**Missing CUB-200 data**: Download from https://data.caltech.edu/records/65de6-vp158
and extract to ./data/CUB_200_2011/.

**CIFAR-100 resolution warning**: This is expected. CIFAR images are 32x32 but VLM
teachers need 224x224. The caching script handles upscaling. Students train at 32x32.
