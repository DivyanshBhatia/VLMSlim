"""
VLMSlim — Analysis & Figure Generation
========================================
Generates all paper figures and tables from experiment logs.

Usage:
    python analyze.py --output_dir ./outputs --figures_dir ./figures
    python analyze.py --output_dir ./outputs --figure hero      # Just hero figure
    python analyze.py --output_dir ./outputs --figure ablation  # Just ablation bar chart
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ──────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────

COLORS = {
    "vlmslim":     "#059669",   # Emerald green
    "concurrent":  "#2563EB",   # Blue
    "naive_seq":   "#DC2626",   # Red
    "single_vlm":  "#7C3AED",   # Purple
    "vanilla_kd":  "#9CA3AF",   # Gray
    "no_anchor":   "#F59E0B",   # Amber
    "no_cumul":    "#06B6D4",   # Cyan
    "no_feature":  "#EC4899",   # Pink
}

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────

def load_experiment(output_dir: str, exp_id: str, dataset: str, student: str,
                     seed: int) -> dict:
    path = os.path.join(output_dir, exp_id, f"{dataset}_{student}_seed{seed}",
                         "experiment_log.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_multi_seed(output_dir: str, exp_id: str, dataset: str, student: str,
                     seeds: list) -> list:
    results = []
    for seed in seeds:
        r = load_experiment(output_dir, exp_id, dataset, student, seed)
        if r:
            results.append(r)
    return results


def get_val_acc_curve(result: dict) -> list:
    """Extract per-epoch validation accuracy."""
    if result and "history" in result and "val_acc" in result["history"]:
        return result["history"]["val_acc"]
    return []


def get_test_acc(result: dict) -> float:
    if result and "metadata" in result and "final_results" in result["metadata"]:
        return result["metadata"]["final_results"]["test_acc"]
    return None


# ──────────────────────────────────────────────────────────
# Figure 2: HERO FIGURE — Training Stability Curves
# ──────────────────────────────────────────────────────────

def plot_hero_figure(output_dir: str, figures_dir: str, dataset: str = "cifar100",
                      student: str = "resnet18", seeds: list = [42]):
    """Plot overlaid training curves: naive seq vs concurrent vs VLMSlim."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [
        ("exp2_naive_seq",         "Naive Sequential",     COLORS["naive_seq"],   "--", 1.5),
        ("exp1_concurrent",        "Concurrent Average",   COLORS["concurrent"],  ":",  1.5),
        ("exp3_vlmslim_lam0.1",    "VLMSlim (ours)",       COLORS["vlmslim"],     "-",  2.5),
    ]

    for exp_id, label, color, linestyle, linewidth in configs:
        all_curves = []
        for seed in seeds:
            result = load_experiment(output_dir, exp_id, dataset, student, seed)
            curve = get_val_acc_curve(result)
            if curve:
                all_curves.append(curve)

        if not all_curves:
            print(f"  [WARN] No data for {exp_id}")
            continue

        # Align lengths
        min_len = min(len(c) for c in all_curves)
        curves = np.array([c[:min_len] for c in all_curves])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)

        epochs = np.arange(min_len)
        ax.plot(epochs, mean, label=label, color=color, linestyle=linestyle,
                linewidth=linewidth)
        if len(all_curves) > 1:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)

    # Phase boundaries
    total_epochs = len(epochs) if 'epochs' in dir() else 240
    K = 3
    phase_len = total_epochs // K
    for i in range(1, K):
        boundary = phase_len * i
        ax.axvline(x=boundary, color="#D1D5DB", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(boundary + 2, ax.get_ylim()[0] + 1, f"Phase {i}→{i+1}",
                fontsize=8, color="#6B7280", rotation=90, va="bottom")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(f"Training Stability — {dataset.upper()} / {student}")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(figures_dir, "fig2_hero_stability_curves.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # Also save PNG for quick viewing
    path_png = path.replace(".pdf", ".png")
    fig, ax = plt.subplots(figsize=(8, 5))
    # Re-plot for PNG (matplotlib figure can't be reused after savefig+close)
    for exp_id, label, color, linestyle, linewidth in configs:
        all_curves = []
        for seed in seeds:
            result = load_experiment(output_dir, exp_id, dataset, student, seed)
            curve = get_val_acc_curve(result)
            if curve:
                all_curves.append(curve)
        if not all_curves:
            continue
        min_len = min(len(c) for c in all_curves)
        curves = np.array([c[:min_len] for c in all_curves])
        mean = curves.mean(axis=0)
        epochs = np.arange(min_len)
        ax.plot(epochs, mean, label=label, color=color, linestyle=linestyle,
                linewidth=linewidth)

    for i in range(1, K):
        ax.axvline(x=phase_len * i, color="#D1D5DB", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(path_png)
    plt.close()
    print(f"  Saved: {path_png}")


# ──────────────────────────────────────────────────────────
# Figure 3: Ablation Bar Chart
# ──────────────────────────────────────────────────────────

def plot_ablation_bars(output_dir: str, figures_dir: str, dataset: str = "cifar100",
                        student: str = "resnet18", seeds: list = [42, 123, 456]):
    """Grouped bar chart of all 8 ablation variants."""
    setup_style()

    variants = [
        ("exp4_a_vanilla_kd",    "Vanilla KD",      COLORS["vanilla_kd"]),
        ("exp4_b_single_vlm",    "Single VLM",       COLORS["single_vlm"]),
        ("exp4_c_concurrent",    "Concurrent",        COLORS["concurrent"]),
        ("exp4_d_naive_seq",     "Naive Seq.",        COLORS["naive_seq"]),
        ("exp4_e_no_anchor",     "w/o Anchor",        COLORS["no_anchor"]),
        ("exp4_f_no_cumulative", "w/o Cumul.",        COLORS["no_cumul"]),
        ("exp4_g_no_feature",    "w/o Feature",       COLORS["no_feature"]),
        ("exp4_h_full",          "VLMSlim\n(full)",   COLORS["vlmslim"]),
    ]

    labels = []
    means = []
    stds = []
    colors = []

    for exp_id, label, color in variants:
        accs = []
        for seed in seeds:
            result = load_experiment(output_dir, exp_id, dataset, student, seed)
            acc = get_test_acc(result)
            if acc is not None:
                accs.append(acc)

        if accs:
            labels.append(label)
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            colors.append(color)
        else:
            labels.append(label)
            means.append(0)
            stds.append(0)
            colors.append("#E5E7EB")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=1.2)

    # Add value labels
    for bar, m, s in zip(bars, means, stds):
        if m > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.3,
                    f"{m:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Ablation Study — {dataset.upper()} / {student}")
    ax.grid(True, axis="y", alpha=0.3)

    # Set y-axis range for better visibility
    if any(m > 0 for m in means):
        valid_means = [m for m in means if m > 0]
        ax.set_ylim(min(valid_means) - 5, max(valid_means) + 5)

    path = os.path.join(figures_dir, "fig3_ablation_bars.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# Figure 4: λ Sensitivity
# ──────────────────────────────────────────────────────────

def plot_lambda_sensitivity(output_dir: str, figures_dir: str, dataset: str = "cifar100",
                             student: str = "resnet18", seeds: list = [42, 123, 456]):
    """Line plot of accuracy vs anchor strength λ."""
    setup_style()

    lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]
    means = []
    stds = []

    for lam in lambdas:
        accs = []
        for seed in seeds:
            result = load_experiment(output_dir, f"lambda_sweep_lam{lam}",
                                      dataset, student, seed)
            acc = get_test_acc(result)
            if acc is not None:
                accs.append(acc)

        if accs:
            means.append(np.mean(accs))
            stds.append(np.std(accs))
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(lambdas, means, yerr=stds, marker="o", color=COLORS["vlmslim"],
                linewidth=2, markersize=8, capsize=5)

    ax.set_xscale("log")
    ax.set_xlabel("Anchor Strength λ")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"λ Sensitivity — {dataset.upper()} / {student}")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.grid(True, alpha=0.3)

    # Highlight best λ
    if any(m > 0 for m in means):
        best_idx = np.argmax(means)
        ax.annotate(f"Best: λ={lambdas[best_idx]}",
                    xy=(lambdas[best_idx], means[best_idx]),
                    xytext=(15, 15), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color=COLORS["vlmslim"]),
                    fontsize=10, fontweight="bold", color=COLORS["vlmslim"])

    path = os.path.join(figures_dir, "fig4_lambda_sensitivity.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# Figure 5: Cross-Modal Awareness (Feature Distance Ratio)
# ──────────────────────────────────────────────────────────

def plot_feature_distance(output_dir: str, figures_dir: str, dataset: str = "cub200",
                           student: str = "resnet18", seeds: list = [42, 123, 456]):
    """Bar chart comparing inter/intra-class distance ratios."""
    setup_style()

    variants = [
        ("exp4_a_vanilla_kd",  "Vanilla KD",    COLORS["vanilla_kd"]),
        ("exp4_b_single_vlm",  "Single VLM",    COLORS["single_vlm"]),
        ("exp4_h_full",        "VLMSlim",        COLORS["vlmslim"]),
    ]

    labels = []
    means = []
    stds = []
    colors = []

    for exp_id, label, color in variants:
        ratios = []
        for seed in seeds:
            result = load_experiment(output_dir, exp_id, dataset, student, seed)
            if result and "metadata" in result and "final_results" in result["metadata"]:
                fr = result["metadata"]["final_results"].get("feature_metrics", {})
                ratio = fr.get("distance_ratio", None)
                if ratio is not None:
                    ratios.append(ratio)

        if ratios:
            labels.append(label)
            means.append(np.mean(ratios))
            stds.append(np.std(ratios))
            colors.append(color)

    if not labels:
        print("  [WARN] No feature distance data available.")
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=1.2, width=0.5)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Inter-class / Intra-class Distance Ratio")
    ax.set_title(f"Feature Space Quality — {dataset.upper()}")
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(figures_dir, "fig5_feature_distance.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# Phase Boundary Sensitivity Table
# ──────────────────────────────────────────────────────────

def print_phase_sensitivity(output_dir: str, dataset: str = "cifar100",
                              student: str = "resnet18", seeds: list = [42, 123, 456]):
    """Print phase boundary sensitivity results."""
    schedules = {
        "equal_80_160": "(80, 160) — equal",
        "early_60_120": "(60, 120) — early",
        "late_100_180": "(100, 180) — late",
        "front_80_180": "(80, 180) — front-loaded",
        "back_60_160": "(60, 160) — back-loaded",
    }

    print(f"\n  Phase Boundary Sensitivity — {dataset.upper()} / {student}")
    print(f"  {'Schedule':<30s}  {'Acc (mean±std)':>16s}")
    print(f"  {'─'*30}  {'─'*16}")

    all_means = []
    for label_key, display in schedules.items():
        accs = []
        for seed in seeds:
            result = load_experiment(output_dir, f"phase_sens_{label_key}",
                                      dataset, student, seed)
            acc = get_test_acc(result)
            if acc is not None:
                accs.append(acc)

        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            all_means.append(mean)
            print(f"  {display:<30s}  {mean:>6.2f} ± {std:.2f}")
        else:
            print(f"  {display:<30s}  {'—':>16s}")

    if len(all_means) >= 2:
        variation = max(all_means) - min(all_means)
        print(f"\n  Max variation: {variation:.2f}%")
        if variation <= 1.5:
            print(f"  ✅ Robust (≤1.5% variation)")
        else:
            print(f"  ⚠  Sensitive (>1.5%) — consider adaptive switching")


# ──────────────────────────────────────────────────────────
# Gradient Norm Analysis (Supplementary)
# ──────────────────────────────────────────────────────────

def plot_gradient_norms(output_dir: str, figures_dir: str, dataset: str = "cifar100",
                         student: str = "resnet18", seed: int = 42):
    """Plot gradient norms over training for naive seq vs VLMSlim."""
    setup_style()

    configs = [
        ("exp2_naive_seq",         "Naive Sequential",  COLORS["naive_seq"],  "--"),
        ("exp3_vlmslim_lam0.1",    "VLMSlim",           COLORS["vlmslim"],    "-"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    for exp_id, label, color, ls in configs:
        result = load_experiment(output_dir, exp_id, dataset, student, seed)
        if result and "history" in result and "grad_norm" in result["history"]:
            norms = result["history"]["grad_norm"]
            ax.plot(norms, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Stability Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(figures_dir, "supp_gradient_norms.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate VLMSlim paper figures")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--figures_dir", type=str, default="./figures")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--figure", type=str, default="all",
                        choices=["all", "hero", "ablation", "lambda", "feature",
                                 "gradient", "phase_table"])
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  VLMSlim Figure Generation")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Figures dir: {args.figures_dir}")
    print(f"{'='*70}\n")

    if args.figure in ("all", "hero"):
        print("  [Figure 2] Hero: Training stability curves")
        plot_hero_figure(args.output_dir, args.figures_dir, args.dataset,
                          args.student, args.seeds)

    if args.figure in ("all", "ablation"):
        print("  [Figure 3] Ablation bar chart")
        plot_ablation_bars(args.output_dir, args.figures_dir, args.dataset,
                            args.student, args.seeds)

    if args.figure in ("all", "lambda"):
        print("  [Figure 4] λ sensitivity")
        plot_lambda_sensitivity(args.output_dir, args.figures_dir, args.dataset,
                                 args.student, args.seeds)

    if args.figure in ("all", "feature"):
        print("  [Figure 5] Feature distance ratio")
        plot_feature_distance(args.output_dir, args.figures_dir, "cub200",
                               args.student, args.seeds)

    if args.figure in ("all", "gradient"):
        print("  [Supp] Gradient norms")
        plot_gradient_norms(args.output_dir, args.figures_dir, args.dataset,
                             args.student, args.seeds[0])

    if args.figure in ("all", "phase_table"):
        print_phase_sensitivity(args.output_dir, args.dataset, args.student, args.seeds)

    print(f"\n  Done. Figures saved to: {args.figures_dir}\n")


if __name__ == "__main__":
    main()
