#!/usr/bin/env python3
"""
VLMSlim — α/β Diagnostic Sweep
================================
Before committing to fixed loss weights, run this quick diagnostic to find
the right balance between CE, KD logits, and feature alignment for your
teacher/dataset combination.

The issue: standard KD uses α=0.1, β=0.9, designed for 95%+ teachers.
Frozen VLM teachers are only 62-73% on CIFAR-100 — their logits are noisy.
This sweep tests whether logits help, hurt, or don't matter compared to
the feature alignment path.

Runs 4 configurations at 50 epochs (not full 240) on a single seed:

    A) feat_only:   α=1.0, β=0.0  — CE + features, no KD logits at all
    B) ce_heavy:    α=0.5, β=0.1  — CE-dominated, KD as mild regularizer
    C) balanced:    α=0.3, β=0.2  — our default (feature-centric)
    D) kd_medium:   α=0.3, β=0.5  — more KD weight — do 73% logits help?

All runs keep feature_weight=0.5 and use_feature_path=True.

Usage:
    python diagnostic_sweep.py --dataset cifar100 --teacher openclip_vitl14
    python diagnostic_sweep.py --dataset cifar100 --teacher clip_vitb16
    python diagnostic_sweep.py --dataset cub200 --teacher openclip_vitl14

Takes ~2 hours total on a single GPU. Prints a comparison table at the end.
"""

import argparse
import json
import os
import sys
import time

from config import ExperimentConfig, DATASETS, TEACHERS
from train import run_experiment


# ──────────────────────────────────────────────────────────
# Diagnostic configurations
# ──────────────────────────────────────────────────────────

SWEEP_CONFIGS = {
    "A_feat_only": dict(
        alpha=1.0, beta=0.0,
        label="A) CE + features only (β=0)",
        hypothesis="If this wins, KD logits from frozen VLMs are pure noise",
    ),
    "B_ce_heavy": dict(
        alpha=0.5, beta=0.1,
        label="B) CE-heavy (α=0.5, β=0.1)",
        hypothesis="KD as very mild regularizer",
    ),
    "C_balanced": dict(
        alpha=0.3, beta=0.2,
        label="C) Balanced (α=0.3, β=0.2)",
        hypothesis="Feature-centric default",
    ),
    "D_kd_medium": dict(
        alpha=0.3, beta=0.5,
        label="D) More KD (α=0.3, β=0.5)",
        hypothesis="Do 73% logits actually help?",
    ),
}

DIAGNOSTIC_EPOCHS = 50


def make_diagnostic_config(
    name: str,
    alpha: float,
    beta: float,
    teacher_key: str,
    dataset: str,
    seed: int,
    **kwargs,
) -> ExperimentConfig:
    """Build a short diagnostic experiment config."""
    return ExperimentConfig(
        exp_name=f"Diagnostic: {name} (α={alpha}, β={beta})",
        exp_id=f"diag_{name}_{teacher_key}",
        seed=seed,
        dataset=dataset,
        student="resnet18",
        teachers=[teacher_key],
        alpha=alpha,
        beta=beta,
        feature_weight=0.5,
        use_cumulative_targets=False,
        use_anchor=False,
        use_feature_path=True,
        sequential=False,
        lam=0.0,
    )


def load_diagnostic_result(output_dir: str, exp_id: str, dataset: str,
                            seed: int) -> dict:
    """Load experiment log JSON."""
    path = os.path.join(
        output_dir, exp_id,
        f"{dataset}_resnet18_seed{seed}",
        "experiment_log.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(result: dict) -> dict:
    """Pull the key comparison metrics from a result."""
    if result is None:
        return None

    meta = result.get("metadata", {})
    final = meta.get("final_results", {})
    hist = result.get("history", {})
    feat = final.get("feature_metrics", {})

    # Get val acc at epoch 50 (or last available)
    val_accs = hist.get("val_acc", [])
    epoch_50_acc = val_accs[49] if len(val_accs) >= 50 else (val_accs[-1] if val_accs else 0)

    # Best val acc across all epochs
    best_val = max(val_accs) if val_accs else 0

    return {
        "test_acc": final.get("test_acc", 0),
        "best_val_acc": final.get("best_val_acc", 0),
        "epoch_50_val": epoch_50_acc,
        "distance_ratio": feat.get("distance_ratio", 0),
        "inter_class_dist": feat.get("inter_class_dist", 0),
        "intra_class_dist": feat.get("intra_class_dist", 0),
        "derived_gamma": meta.get("derived_gamma", None),
    }


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLMSlim α/β diagnostic sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python diagnostic_sweep.py --dataset cifar100 --teacher openclip_vitl14
    python diagnostic_sweep.py --dataset cifar100 --teacher clip_vitb16
    python diagnostic_sweep.py --dataset cub200   --teacher openclip_vitl14
    python diagnostic_sweep.py --dataset cifar100 --teacher openclip_vitl14 --epochs 100
        """,
    )
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cub200"])
    parser.add_argument("--teacher", type=str, default="openclip_vitl14",
                        help="Single VLM teacher to test with")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DIAGNOSTIC_EPOCHS,
                        help="Epochs per run (default: 50, enough to see trends)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=list(SWEEP_CONFIGS.keys()),
                        help="Which configs to run (default: all four)")
    args = parser.parse_args()

    if args.teacher not in TEACHERS:
        print(f"  [ERROR] Unknown teacher: {args.teacher}")
        print(f"  Available: {list(TEACHERS.keys())}")
        sys.exit(1)

    if TEACHERS[args.teacher].requires_finetune:
        print(f"  [ERROR] {args.teacher} is a vision-only teacher.")
        print(f"  This diagnostic is for frozen VLM teachers only.")
        sys.exit(1)

    # Override dataset epochs for quick runs
    original_epochs = DATASETS[args.dataset].total_epochs
    DATASETS[args.dataset].total_epochs = args.epochs
    DATASETS[args.dataset].data_root = args.data_root

    print(f"\n{'█'*70}")
    print(f"  VLMSlim α/β Diagnostic Sweep")
    print(f"  Teacher:  {args.teacher}")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Epochs:   {args.epochs} (quick diagnostic, not full training)")
    print(f"  Seed:     {args.seed}")
    print(f"  Configs:  {args.configs}")
    print(f"{'█'*70}\n")

    total_start = time.time()
    results = {}

    for config_name in args.configs:
        if config_name not in SWEEP_CONFIGS:
            print(f"  [SKIP] Unknown config: {config_name}")
            continue

        sweep = SWEEP_CONFIGS[config_name]
        print(f"\n{'▓'*70}")
        print(f"  {sweep['label']}")
        print(f"  Hypothesis: {sweep['hypothesis']}")
        print(f"{'▓'*70}")

        cfg = make_diagnostic_config(
            name=config_name,
            alpha=sweep["alpha"],
            beta=sweep["beta"],
            teacher_key=args.teacher,
            dataset=args.dataset,
            seed=args.seed,
        )
        cfg.cache_dir = args.cache_dir
        cfg.output_dir = args.output_dir

        run_experiment(cfg)

        # Load results
        result = load_diagnostic_result(
            args.output_dir, cfg.exp_id, args.dataset, args.seed
        )
        metrics = extract_metrics(result)
        if metrics:
            results[config_name] = metrics

    # Restore original epochs
    DATASETS[args.dataset].total_epochs = original_epochs

    # ──────────────────────────────────────────────────────────
    # Print comparison table
    # ──────────────────────────────────────────────────────────

    total_time = time.time() - total_start

    print(f"\n\n{'█'*70}")
    print(f"  DIAGNOSTIC RESULTS — {args.teacher} on {args.dataset}")
    print(f"  ({args.epochs} epochs, seed {args.seed}, {total_time/60:.0f} min total)")
    print(f"{'█'*70}\n")

    if not results:
        print("  No results collected. Check for errors above.")
        return

    # Header
    print(f"  {'Config':<32s}  {'α':>4s}  {'β':>4s}  "
          f"{'Val@{0}'.format(args.epochs):>7s}  {'Test':>6s}  "
          f"{'Feat Ratio':>10s}  {'γ':>8s}")
    print(f"  {'─'*32}  {'─'*4}  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*10}  {'─'*8}")

    best_ratio = 0
    best_ratio_name = ""
    best_acc = 0
    best_acc_name = ""

    for config_name in args.configs:
        if config_name not in results:
            continue
        m = results[config_name]
        sweep = SWEEP_CONFIGS[config_name]

        gamma_str = f"{m['derived_gamma']:.1f}" if m['derived_gamma'] else "—"
        print(f"  {sweep['label']:<32s}  {sweep['alpha']:>4.1f}  {sweep['beta']:>4.1f}  "
              f"{m['epoch_50_val']:>6.2f}%  {m['test_acc']:>5.2f}%  "
              f"{m['distance_ratio']:>10.4f}  {gamma_str:>8s}")

        if m['distance_ratio'] > best_ratio:
            best_ratio = m['distance_ratio']
            best_ratio_name = config_name
        if m['test_acc'] > best_acc:
            best_acc = m['test_acc']
            best_acc_name = config_name

    # Interpretation
    print(f"\n  {'─'*70}")
    print(f"  Best accuracy:       {SWEEP_CONFIGS[best_acc_name]['label']} ({best_acc:.2f}%)")
    print(f"  Best feature ratio:  {SWEEP_CONFIGS[best_ratio_name]['label']} ({best_ratio:.4f})")

    # Decision logic
    a_metrics = results.get("A_feat_only")
    c_metrics = results.get("C_balanced")
    d_metrics = results.get("D_kd_medium")

    print(f"\n  INTERPRETATION:")
    if a_metrics and c_metrics:
        acc_diff = c_metrics["test_acc"] - a_metrics["test_acc"]
        ratio_diff = c_metrics["distance_ratio"] - a_metrics["distance_ratio"]

        if acc_diff < 0.5 and ratio_diff < 0.01:
            print(f"    → KD logits don't help. Set β=0.0 (features + CE only).")
            print(f"      C vs A: acc {acc_diff:+.2f}%, ratio {ratio_diff:+.4f}")
        elif acc_diff > 0.5:
            print(f"    → KD logits help accuracy. Keep β>0.")
            print(f"      C vs A: acc {acc_diff:+.2f}%, ratio {ratio_diff:+.4f}")
        else:
            print(f"    → KD logits don't help accuracy but may affect features.")
            print(f"      C vs A: acc {acc_diff:+.2f}%, ratio {ratio_diff:+.4f}")

    if c_metrics and d_metrics:
        acc_diff = d_metrics["test_acc"] - c_metrics["test_acc"]
        if acc_diff > 0.5:
            print(f"    → Higher β helps. Consider β=0.5 over β=0.2.")
            print(f"      D vs C: acc {acc_diff:+.2f}%")
        elif acc_diff < -0.5:
            print(f"    → Higher β hurts. β=0.2 or lower is correct.")
            print(f"      D vs C: acc {acc_diff:+.2f}%")
        else:
            print(f"    → β=0.2 vs β=0.5 is within noise. Either works.")
            print(f"      D vs C: acc {acc_diff:+.2f}%")

    print(f"\n  NEXT STEP:")
    print(f"    Pick the best config, update α/β in config.py, and run full Exp 0:")
    print(f"    python run_experiments.py --exp exp0 --seeds 42 123 456")

    # Save summary
    summary_path = os.path.join(
        args.output_dir,
        f"diagnostic_{args.teacher}_{args.dataset}_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump({
            "teacher": args.teacher,
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "total_time_minutes": total_time / 60,
            "results": results,
            "best_accuracy": {"config": best_acc_name, "value": best_acc},
            "best_feature_ratio": {"config": best_ratio_name, "value": best_ratio},
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    print()


if __name__ == "__main__":
    main()
