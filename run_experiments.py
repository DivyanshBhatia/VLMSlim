#!/usr/bin/env python3
"""
VLMSlim — Experiment Runner
=============================
Orchestrates all experiments in priority order with go/no-go gates.

Usage:
    # Step 0: Cache teacher outputs first (run once per dataset)
    python cache_teachers.py --dataset cifar100 --data_root ./data --cache_dir ./cache

    # Then run experiments:
    python run_experiments.py --exp all            # Run everything in order
    python run_experiments.py --exp exp0            # Just sanity check
    python run_experiments.py --exp exp1            # Concurrent baseline
    python run_experiments.py --exp exp2            # Naive sequential
    python run_experiments.py --exp exp3            # Full VLMSlim
    python run_experiments.py --exp exp4            # Ablation table
    python run_experiments.py --exp exp5            # CUB-200 + λ sweep
    python run_experiments.py --exp exp6            # ImageNet (parallel)
    python run_experiments.py --exp lambda_sweep    # Just the λ sweep

    # Multi-seed runs:
    python run_experiments.py --exp exp3 --seeds 42 123 456

    # Custom settings:
    python run_experiments.py --exp exp3 --dataset cub200 --student mobilenetv2
"""

import argparse
import json
import os
import sys
from typing import List, Dict

from config import (
    ExperimentConfig, DATASETS,
    exp0_scratch, exp0_sanity, exp1_concurrent, exp2_naive_sequential,
    exp3_vlmslim, exp4_ablation,
)
from train import run_experiment


# ──────────────────────────────────────────────────────────
# Gate checking
# ──────────────────────────────────────────────────────────

def load_result(output_dir: str, exp_id: str, dataset: str, student: str,
                seed: int) -> Dict:
    """Load experiment results from JSON log."""
    path = os.path.join(
        output_dir, exp_id,
        f"{dataset}_{student}_seed{seed}",
        "experiment_log.json"
    )
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_test_acc(result: Dict) -> float:
    """Extract test accuracy from result dict."""
    if result and "metadata" in result and "final_results" in result["metadata"]:
        return result["metadata"]["final_results"]["test_acc"]
    return None


def get_feature_ratio(result: Dict) -> float:
    """Extract inter/intra-class distance ratio from result dict."""
    if result and "metadata" in result and "final_results" in result["metadata"]:
        fm = result["metadata"]["final_results"].get("feature_metrics", {})
        return fm.get("distance_ratio", None)
    return None


def check_gate_exp0(output_dir: str, dataset: str, student: str, seeds: List[int],
                     feature_threshold: float = 0.10) -> bool:
    """Gate: VLM distillation must produce better feature structure than scratch.

    Dual check:
    1. Feature quality (primary): VLM student's inter/intra distance ratio
       must exceed scratch by ≥feature_threshold (default 10%)
    2. Accuracy (informational): reported but NOT a kill switch.

    Rationale: frozen VLM teachers have weak zero-shot logits (62-73%)
    but geometrically rich feature spaces. The value of VLM distillation
    is in transferring that feature structure, not in matching accuracy
    of a scratch-trained model.
    """
    scratch_accs = []
    scratch_ratios = []
    vlm_accs = []
    vlm_ratios = []

    for seed in seeds:
        scratch = load_result(output_dir, "exp0_scratch", dataset, student, seed)
        vlm = load_result(output_dir, "exp0_openclip_vitl14", dataset, student, seed)

        s_acc = get_test_acc(scratch)
        s_ratio = get_feature_ratio(scratch)
        v_acc = get_test_acc(vlm)
        v_ratio = get_feature_ratio(vlm)

        if s_acc is not None:
            scratch_accs.append(s_acc)
        if s_ratio is not None:
            scratch_ratios.append(s_ratio)
        if v_acc is not None:
            vlm_accs.append(v_acc)
        if v_ratio is not None:
            vlm_ratios.append(v_ratio)

    print(f"\n  [GATE] Exp 0 — Cross-Modal Feature Transfer:")

    # Accuracy (informational)
    if scratch_accs and vlm_accs:
        s_mean_acc = sum(scratch_accs) / len(scratch_accs)
        v_mean_acc = sum(vlm_accs) / len(vlm_accs)
        acc_gap = v_mean_acc - s_mean_acc
        print(f"    Accuracy:")
        print(f"      Scratch baseline:       {s_mean_acc:.2f}%")
        print(f"      VLM (OpenCLIP):         {v_mean_acc:.2f}%")
        print(f"      Gap:                    {acc_gap:+.2f}%  (informational, not gated)")

    # Feature quality (primary gate)
    if scratch_ratios and vlm_ratios:
        s_mean_ratio = sum(scratch_ratios) / len(scratch_ratios)
        v_mean_ratio = sum(vlm_ratios) / len(vlm_ratios)
        ratio_improvement = (v_mean_ratio - s_mean_ratio) / max(s_mean_ratio, 1e-8)

        print(f"    Feature distance ratio (inter/intra):")
        print(f"      Scratch baseline:       {s_mean_ratio:.4f}")
        print(f"      VLM (OpenCLIP):         {v_mean_ratio:.4f}")
        print(f"      Improvement:            {ratio_improvement*100:+.1f}%")
        print(f"      Required:               ≥{feature_threshold*100:.0f}%")

        if ratio_improvement >= feature_threshold:
            print(f"    ✅ PASS — VLM produces richer feature geometry.")
            return True
        else:
            print(f"    ❌ FAIL — VLM feature structure not significantly better.")
            print(f"    ⚠  Cross-modal feature transfer not confirmed.")
            return False
    else:
        print(f"    [GATE] Insufficient feature metrics. Check that feature collection is enabled.")
        return True  # Proceed with caution


def check_gate_exp2(output_dir: str, dataset: str, student: str,
                     seeds: List[int]) -> bool:
    """Gate: Naive sequential must underperform single-teacher (OpenCLIP)."""
    naive_accs = []
    single_accs = []

    for seed in seeds:
        naive = load_result(output_dir, "exp2_naive_seq", dataset, student, seed)
        single = load_result(output_dir, "exp0_openclip_vitl14", dataset, student, seed)

        n_acc = get_test_acc(naive)
        s_acc = get_test_acc(single)

        if n_acc is not None:
            naive_accs.append(n_acc)
        if s_acc is not None:
            single_accs.append(s_acc)

    if not naive_accs or not single_accs:
        print("  [GATE] Exp 2: Insufficient results.")
        return True

    naive_mean = sum(naive_accs) / len(naive_accs)
    single_mean = sum(single_accs) / len(single_accs)

    print(f"\n  [GATE] Exp 2 — Moving-Goal Problem:")
    print(f"    Naive sequential:         {naive_mean:.2f}%")
    print(f"    Single VLM (OpenCLIP):    {single_mean:.2f}%")
    print(f"    Expected: naive < single")

    if naive_mean < single_mean:
        print(f"    ✅ PASS — Moving-goal problem demonstrated ({single_mean - naive_mean:.2f}% gap).")
        return True
    else:
        print(f"    ❌ FAIL — Naive sequential did NOT underperform.")
        print(f"    ⚠  Moving-goal problem may not be real. Reframe paper.")
        return False


def check_gate_exp3(output_dir: str, dataset: str, student: str,
                     seeds: List[int]) -> bool:
    """Gate: VLMSlim must beat concurrent average."""
    vlmslim_accs = []
    concurrent_accs = []

    for seed in seeds:
        vlmslim = load_result(output_dir, "exp3_vlmslim_lam0.1", dataset, student, seed)
        concurrent = load_result(output_dir, "exp1_concurrent", dataset, student, seed)

        v_acc = get_test_acc(vlmslim)
        c_acc = get_test_acc(concurrent)

        if v_acc is not None:
            vlmslim_accs.append(v_acc)
        if c_acc is not None:
            concurrent_accs.append(c_acc)

    if not vlmslim_accs or not concurrent_accs:
        print("  [GATE] Exp 3: Insufficient results.")
        return True

    vlmslim_mean = sum(vlmslim_accs) / len(vlmslim_accs)
    concurrent_mean = sum(concurrent_accs) / len(concurrent_accs)

    print(f"\n  [GATE] Exp 3 — VLMSlim vs Concurrent Average:")
    print(f"    VLMSlim:                  {vlmslim_mean:.2f}%")
    print(f"    Concurrent average:       {concurrent_mean:.2f}%")

    if vlmslim_mean > concurrent_mean:
        print(f"    ✅ PASS — VLMSlim beats concurrent by {vlmslim_mean - concurrent_mean:.2f}%.")
        return True
    else:
        print(f"    ❌ FAIL — Sequential scheduling not helping.")
        print(f"    ⚠  KILL SWITCH: Paper likely not viable for BMVC.")
        return False


# ──────────────────────────────────────────────────────────
# Experiment executors
# ──────────────────────────────────────────────────────────

def run_exp0(seeds: List[int], dataset: str, student: str, **kwargs):
    """Exp 0: Sanity check — scratch baseline vs VLM distillation.

    Runs:
    1. Scratch baseline (pure CE, no teacher) — establishes accuracy ceiling
    2. Single VLM teacher KD (OpenCLIP, CLIP) — tests feature transfer

    Gate checks FEATURE QUALITY, not just accuracy. The VLM student
    must produce a meaningfully higher inter/intra distance ratio.
    """
    print("\n" + "▓" * 70)
    print("  EXP 0 — SANITY CHECK: Scratch vs VLM Feature Transfer")
    print("▓" * 70)

    # A) Scratch baseline
    for seed in seeds:
        cfg = exp0_scratch(seed)
        cfg.dataset = dataset
        cfg.student = student
        cfg.cache_dir = kwargs.get("cache_dir", "./cache")
        cfg.output_dir = kwargs.get("output_dir", "./outputs")
        cfg.use_wandb = kwargs.get("use_wandb", False)
        run_experiment(cfg)

    # B) VLM teachers (frozen)
    vlm_teachers = ["openclip_vitl14", "clip_vitb16"]
    for teacher_key in vlm_teachers:
        for seed in seeds:
            cfg = exp0_sanity(teacher_key, seed)
            cfg.dataset = dataset
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)

    return check_gate_exp0(kwargs.get("output_dir", "./outputs"), dataset, student, seeds)


def run_exp1(seeds: List[int], dataset: str, student: str, **kwargs):
    """Exp 1: Concurrent weighted-average baseline."""
    print("\n" + "▓" * 70)
    print("  EXP 1 — CONCURRENT WEIGHTED-AVERAGE BASELINE")
    print("▓" * 70)

    for seed in seeds:
        cfg = exp1_concurrent(seed)
        cfg.dataset = dataset
        cfg.student = student
        cfg.cache_dir = kwargs.get("cache_dir", "./cache")
        cfg.output_dir = kwargs.get("output_dir", "./outputs")
        cfg.use_wandb = kwargs.get("use_wandb", False)
        run_experiment(cfg)


def run_exp2(seeds: List[int], dataset: str, student: str, **kwargs):
    """Exp 2: Naive sequential (no stability mechanisms)."""
    print("\n" + "▓" * 70)
    print("  EXP 2 — NAIVE SEQUENTIAL (NO STABILITY)")
    print("▓" * 70)

    for seed in seeds:
        cfg = exp2_naive_sequential(seed)
        cfg.dataset = dataset
        cfg.student = student
        cfg.cache_dir = kwargs.get("cache_dir", "./cache")
        cfg.output_dir = kwargs.get("output_dir", "./outputs")
        cfg.use_wandb = kwargs.get("use_wandb", False)
        run_experiment(cfg)

    return check_gate_exp2(kwargs.get("output_dir", "./outputs"), dataset, student, seeds)


def run_exp3(seeds: List[int], dataset: str, student: str, lam: float = 0.1, **kwargs):
    """Exp 3: Full VLMSlim."""
    print("\n" + "▓" * 70)
    print("  EXP 3 — FULL VLMSlim")
    print("▓" * 70)

    for seed in seeds:
        cfg = exp3_vlmslim(seed, lam=lam)
        cfg.dataset = dataset
        cfg.student = student
        cfg.cache_dir = kwargs.get("cache_dir", "./cache")
        cfg.output_dir = kwargs.get("output_dir", "./outputs")
        cfg.use_wandb = kwargs.get("use_wandb", False)
        run_experiment(cfg)

    return check_gate_exp3(kwargs.get("output_dir", "./outputs"), dataset, student, seeds)


def run_exp4(seeds: List[int], dataset: str, student: str, **kwargs):
    """Exp 4: Ablation table — all 8 variants."""
    print("\n" + "▓" * 70)
    print("  EXP 4 — ABLATION TABLE")
    print("▓" * 70)

    variants = [
        "a_vanilla_kd", "b_single_vlm", "c_concurrent", "d_naive_seq",
        "e_no_anchor", "f_no_cumulative", "g_no_feature", "h_full",
    ]

    for variant in variants:
        for seed in seeds:
            cfg = exp4_ablation(variant, seed)
            cfg.dataset = dataset
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)

    # Print summary table
    print_ablation_summary(kwargs.get("output_dir", "./outputs"), dataset, student, seeds)


def run_lambda_sweep(seeds: List[int], dataset: str, student: str, **kwargs):
    """λ sensitivity sweep."""
    print("\n" + "▓" * 70)
    print("  λ SENSITIVITY SWEEP")
    print("▓" * 70)

    lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
    for lam in lambdas:
        for seed in seeds:
            cfg = exp3_vlmslim(seed, lam=lam)
            cfg.exp_id = f"lambda_sweep_lam{lam}"
            cfg.dataset = dataset
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)


def run_phase_sensitivity(seeds: List[int], dataset: str, student: str, **kwargs):
    """Phase boundary robustness check."""
    print("\n" + "▓" * 70)
    print("  PHASE BOUNDARY SENSITIVITY")
    print("▓" * 70)

    schedules = {
        "equal_80_160": "equal",
        "early_60_120": "60,120",
        "late_100_180": "100,180",
        "front_80_180": "80,180",
        "back_60_160": "60,160",
    }

    for label, schedule in schedules.items():
        for seed in seeds:
            cfg = exp3_vlmslim(seed, lam=0.1)
            cfg.exp_id = f"phase_sens_{label}"
            cfg.phase_schedule = schedule
            cfg.dataset = dataset
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)


def run_teacher_ordering(seeds: List[int], dataset: str, student: str, **kwargs):
    """Teacher ordering sensitivity."""
    print("\n" + "▓" * 70)
    print("  TEACHER ORDERING SENSITIVITY")
    print("▓" * 70)

    orderings = {
        "best_first": ["openclip_vitl14", "metaclip_vitb16", "clip_vitb16"],
        "worst_first": ["clip_vitb16", "metaclip_vitb16", "openclip_vitl14"],
        "random": ["metaclip_vitb16", "openclip_vitl14", "clip_vitb16"],
    }

    for label, order in orderings.items():
        for seed in seeds:
            cfg = exp3_vlmslim(seed, lam=0.1)
            cfg.exp_id = f"ordering_{label}"
            cfg.teachers = order
            cfg.dataset = dataset
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)


def run_exp5(seeds: List[int], **kwargs):
    """Exp 5: CUB-200 + sensitivity experiments."""
    print("\n" + "▓" * 70)
    print("  EXP 5 — CUB-200 + SENSITIVITY EXPERIMENTS")
    print("▓" * 70)

    student = "resnet18"

    # A) λ sweep on CIFAR-100
    run_lambda_sweep(seeds, "cifar100", student, **kwargs)

    # B) Phase boundary sensitivity on CIFAR-100
    run_phase_sensitivity(seeds, "cifar100", student, **kwargs)

    # C) Teacher ordering on CIFAR-100
    run_teacher_ordering(seeds, "cifar100", student, **kwargs)

    # D) CUB-200 main experiments
    for variant in ["a_vanilla_kd", "b_single_vlm", "c_concurrent", "d_naive_seq", "h_full"]:
        for seed in seeds:
            cfg = exp4_ablation(variant, seed)
            cfg.dataset = "cub200"
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)


def run_exp6(seeds: List[int], **kwargs):
    """Exp 6: ImageNet — 3 key configs with DDP."""
    print("\n" + "▓" * 70)
    print("  EXP 6 — IMAGENET-1K")
    print("▓" * 70)

    student = "resnet18"
    # Run only the most important configs on ImageNet
    for seed in seeds[:1]:  # Single seed for initial ImageNet run
        for variant in ["b_single_vlm", "c_concurrent", "h_full"]:
            cfg = exp4_ablation(variant, seed)
            cfg.dataset = "imagenet"
            cfg.student = student
            cfg.cache_dir = kwargs.get("cache_dir", "./cache")
            cfg.output_dir = kwargs.get("output_dir", "./outputs")
            cfg.use_wandb = kwargs.get("use_wandb", False)
            run_experiment(cfg)


# ──────────────────────────────────────────────────────────
# Summary printing
# ──────────────────────────────────────────────────────────

def print_ablation_summary(output_dir: str, dataset: str, student: str,
                            seeds: List[int]):
    """Print the ablation table from saved results."""
    variants = [
        ("a_vanilla_kd",    "a) Vanilla KD (vision teacher)"),
        ("b_single_vlm",    "b) Single VLM (OpenCLIP)"),
        ("c_concurrent",    "c) Concurrent average (3 VLMs)"),
        ("d_naive_seq",     "d) Naive sequential (3 VLMs)"),
        ("e_no_anchor",     "e) VLMSlim w/o anchor (λ=0)"),
        ("f_no_cumulative", "f) VLMSlim w/o cumulative"),
        ("g_no_feature",    "g) VLMSlim w/o feature path"),
        ("h_full",          "h) VLMSlim (full)"),
    ]

    print(f"\n{'='*70}")
    print(f"  ABLATION TABLE — {dataset} / {student}")
    print(f"{'='*70}")
    print(f"  {'Variant':<40s}  {'Acc (mean±std)':>16s}")
    print(f"  {'─'*40}  {'─'*16}")

    for exp_suffix, label in variants:
        accs = []
        for seed in seeds:
            result = load_result(output_dir, f"exp4_{exp_suffix}", dataset, student, seed)
            acc = get_test_acc(result)
            if acc is not None:
                accs.append(acc)

        if accs:
            import numpy as np
            mean = np.mean(accs)
            std = np.std(accs)
            print(f"  {label:<40s}  {mean:>6.2f} ± {std:.2f}")
        else:
            print(f"  {label:<40s}  {'—':>16s}")

    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLMSlim Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --exp exp0                          # Sanity check only
  python run_experiments.py --exp exp3 --seeds 42 123 456       # VLMSlim, 3 seeds
  python run_experiments.py --exp all                           # Full pipeline
  python run_experiments.py --exp lambda_sweep                  # λ sensitivity
  python run_experiments.py --exp exp4 --dataset cub200         # Ablations on CUB-200
        """
    )
    parser.add_argument("--exp", type=str, required=True,
                        choices=["all", "exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6",
                                 "lambda_sweep", "phase_sensitivity", "teacher_ordering"],
                        help="Which experiment(s) to run")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        choices=["cifar100", "cub200", "imagenet"])
    parser.add_argument("--student", type=str, default="resnet18",
                        choices=["resnet18", "mobilenetv2", "efficientnet_b0"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--lam", type=float, default=0.1,
                        help="λ value for exp3 (default: 0.1)")
    parser.add_argument("--skip_gates", action="store_true",
                        help="Skip go/no-go gates (proceed regardless of results)")

    args = parser.parse_args()

    # Update dataset data_root
    DATASETS[args.dataset].data_root = args.data_root

    kwargs = {
        "cache_dir": args.cache_dir,
        "output_dir": args.output_dir,
        "use_wandb": args.use_wandb,
    }

    print(f"\n{'█'*70}")
    print(f"  VLMSlim Experiment Runner")
    print(f"  Dataset: {args.dataset} | Student: {args.student}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Output: {args.output_dir}")
    print(f"{'█'*70}")

    if args.exp == "all":
        # Full pipeline with gates
        gate_pass = run_exp0(args.seeds, args.dataset, args.student, **kwargs)
        if not gate_pass and not args.skip_gates:
            print("\n  ⛔ STOPPING: Exp 0 gate failed. Use --skip_gates to override.")
            sys.exit(1)

        run_exp1(args.seeds, args.dataset, args.student, **kwargs)

        gate_pass = run_exp2(args.seeds, args.dataset, args.student, **kwargs)
        if not gate_pass and not args.skip_gates:
            print("\n  ⛔ STOPPING: Exp 2 gate failed. Use --skip_gates to override.")
            sys.exit(1)

        gate_pass = run_exp3(args.seeds, args.dataset, args.student, lam=args.lam, **kwargs)
        if not gate_pass and not args.skip_gates:
            print("\n  ⛔ STOPPING: Exp 3 gate failed. Use --skip_gates to override.")
            sys.exit(1)

        run_exp4(args.seeds, args.dataset, args.student, **kwargs)
        run_exp5(args.seeds, **kwargs)

    elif args.exp == "exp0":
        run_exp0(args.seeds, args.dataset, args.student, **kwargs)
    elif args.exp == "exp1":
        run_exp1(args.seeds, args.dataset, args.student, **kwargs)
    elif args.exp == "exp2":
        run_exp2(args.seeds, args.dataset, args.student, **kwargs)
    elif args.exp == "exp3":
        run_exp3(args.seeds, args.dataset, args.student, lam=args.lam, **kwargs)
    elif args.exp == "exp4":
        run_exp4(args.seeds, args.dataset, args.student, **kwargs)
    elif args.exp == "exp5":
        run_exp5(args.seeds, **kwargs)
    elif args.exp == "exp6":
        run_exp6(args.seeds, **kwargs)
    elif args.exp == "lambda_sweep":
        run_lambda_sweep(args.seeds, args.dataset, args.student, **kwargs)
    elif args.exp == "phase_sensitivity":
        run_phase_sensitivity(args.seeds, args.dataset, args.student, **kwargs)
    elif args.exp == "teacher_ordering":
        run_teacher_ordering(args.seeds, args.dataset, args.student, **kwargs)

    print(f"\n{'█'*70}")
    print(f"  ALL REQUESTED EXPERIMENTS COMPLETE")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()
