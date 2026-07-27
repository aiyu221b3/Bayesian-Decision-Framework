"""
validation set ablation study:
evaluates four configurations:
1. Baseline (current): t_min=0, tau=0.50
2. Baseline + t_min=15: t_min=15, tau=0.50
3. Baseline + higher tau: t_min=0, tau=0.75
4. Both together: t_min=15, tau=0.75

compares accuracy, average stopping frame, emissions, and bocpd behaviour.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from changepoint import BOCPD
from policies import BayesThresh, BocpdThresh, RawThresh
from style import use as set_style
from tree_belief import TreeBel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "data" / "all_validation_trays.npz" if (ROOT / "data" / "all_validation_trays.npz").exists() else ROOT / "all_validation_trays.npz"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run_ablation_for_tray(
    P: np.ndarray,
    ground_truth: int,
    t_min: int,
    tau: float,
    K: int = 5,
) -> dict[str, Any]:
    """run bayesian policies under specified t_min and tau constraints."""
    T, _ = P.shape
    S = np.eye(K, dtype=float)
    low = max(0.0, tau - 0.15)

    pols = {
        "M0": RawThresh(K, theta=tau, low=low),
        "M1": BayesThresh(K, lam=0.5, rho=0.0, tau=tau, low=low),
        "M2": BayesThresh(K, lam=0.5, rho=0.05, tau=tau, low=low),
        "M3": BocpdThresh(K, lam=0.5, tau=tau, cp_thr=0.1, low=low),
        "MT": TreeBel(K, S, lam=0.5, rho=0.05, tau=tau, min_sp=0.25, low=low),
    }

    # bocpd changepoints
    cpd = BOCPD(K, hazard=1 / 30.0, alpha=1.0)
    cpd.reset()
    cp_events = []
    for t in range(T):
        snap = cpd.update(int(P[t].argmax()))
        if snap.cp_posterior > 0.1:
            cp_events.append(t)

    result = {
        "ground_truth": int(ground_truth),
        "bocpd_total_detections": len(cp_events),
        "bocpd_after_tmin_detections": sum(1 for c in cp_events if c >= t_min),
        "bocpd_frames": cp_events,
        "policies": {},
    }

    for name, pol in pols.items():
        em = pol.run(P)
        em_valid = [(t, y) for t, y in em if t >= t_min]
        if em_valid:
            t_stop, y_stop = em_valid[0]
            pred = list(y_stop)[0] if isinstance(y_stop, (set, frozenset)) else int(y_stop)
            is_correct = (pred == ground_truth)
            stop_frame = int(t_stop)
            n_emit = len(em_valid)
        else:
            stop_frame = int(T)
            n_emit = 0
            # policy failed to emit; do not credit with final-frame accuracy
            is_correct = False

        result["policies"][name] = {
            "correct": int(is_correct),
            "stopping_frame": stop_frame,
            "n_emissions": n_emit,
            "emitted": int(n_emit > 0),
        }

    return result


def generate_ablation_plots(summary: dict[str, Any], out_dir: Path) -> None:
    """generate grouped bar charts comparing accuracy and stopping frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = list(summary.keys())
    policies = ["M0", "M1", "M2", "M3", "MT"]

    # 1. accuracy comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(policies))
    width = 0.20
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    for i, cfg in enumerate(configs):
        accs = [summary[cfg]["policies"][p]["accuracy_pct"] for p in policies]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, accs, width, label=cfg, color=colors[i], edgecolor="black", linewidth=0.7)
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 1.0,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title("Ablation Study: Validation Set Accuracy (%) by Policy & Configuration", fontsize=13, fontweight="bold")
    ax.set_xlabel("Stopping Policy", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(title="Configuration", frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    plt.savefig(out_dir / "val_ablation_accuracy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. stopping frame comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, cfg in enumerate(configs):
        stops = [summary[cfg]["policies"][p]["mean_stopping_frame"] for p in policies]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, stops, width, label=cfg, color=colors[i], edgecolor="black", linewidth=0.7)
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + 0.8,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("Ablation Study: Average Stopping Frame by Policy & Configuration (Mean Video T=55.6)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Stopping Policy", fontsize=11)
    ax.set_ylabel("Average Stopping Frame", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontweight="bold")
    ax.set_ylim(0, 65)
    ax.legend(title="Configuration", frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    plt.savefig(out_dir / "val_ablation_stopping_frames.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="run ablation study on real validation trays")
    parser.add_argument("--npz", type=str, default=str(DEFAULT_NPZ))
    args = parser.parse_args()

    set_style()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"cannot find real validation npz at {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    trays = data["trays"]
    labels = data["labels"]
    n_seq = len(trays)
    print(f"loaded {n_seq} real validation sequences from {npz_path}")

    configs = {
        "Baseline (t_min=0, tau=0.50)": {"t_min": 0, "tau": 0.50},
        "Baseline + t_min=15 (t_min=15, tau=0.50)": {"t_min": 15, "tau": 0.50},
        "Baseline + higher tau (t_min=0, tau=0.75)": {"t_min": 0, "tau": 0.75},
        "Both together (t_min=15, tau=0.75)": {"t_min": 15, "tau": 0.75},
    }

    summary = {}
    csv_rows = []

    for cfg_name, params in configs.items():
        t_min = params["t_min"]
        tau = params["tau"]
        pol_stats = {
            "M0": {"correct": 0, "stop_sum": 0.0, "emit_sum": 0.0, "emit_count": 0},
            "M1": {"correct": 0, "stop_sum": 0.0, "emit_sum": 0.0, "emit_count": 0},
            "M2": {"correct": 0, "stop_sum": 0.0, "emit_sum": 0.0, "emit_count": 0},
            "M3": {"correct": 0, "stop_sum": 0.0, "emit_sum": 0.0, "emit_count": 0},
            "MT": {"correct": 0, "stop_sum": 0.0, "emit_sum": 0.0, "emit_count": 0},
        }
        bocpd_total = 0
        bocpd_after_tmin = 0

        for idx, (P, gt) in enumerate(zip(trays, labels)):
            res = run_ablation_for_tray(P, int(gt), t_min=t_min, tau=tau)
            bocpd_total += res["bocpd_total_detections"]
            bocpd_after_tmin += res["bocpd_after_tmin_detections"]

            row = {
                "config": cfg_name,
                "seq_idx": idx,
                "ground_truth": int(gt),
                "bocpd_detections": res["bocpd_total_detections"],
            }
            for p in ["M0", "M1", "M2", "M3", "MT"]:
                pol_res = res["policies"][p]
                pol_stats[p]["correct"] += pol_res["correct"]
                pol_stats[p]["stop_sum"] += pol_res["stopping_frame"]
                pol_stats[p]["emit_sum"] += pol_res["n_emissions"]
                pol_stats[p]["emit_count"] += pol_res["emitted"]

                row[f"{p}_correct"] = pol_res["correct"]
                row[f"{p}_stopping_frame"] = pol_res["stopping_frame"]
                row[f"{p}_emissions"] = pol_res["n_emissions"]
            csv_rows.append(row)

        summary[cfg_name] = {
            "t_min": t_min,
            "tau": tau,
            "bocpd_avg_detections": bocpd_total / n_seq,
            "bocpd_avg_after_tmin": bocpd_after_tmin / n_seq,
            "policies": {},
        }

        for p in ["M0", "M1", "M2", "M3", "MT"]:
            summary[cfg_name]["policies"][p] = {
                "accuracy_pct": round((pol_stats[p]["correct"] / n_seq) * 100.0, 1),
                "correct": pol_stats[p]["correct"],
                "total": n_seq,
                "mean_stopping_frame": round(pol_stats[p]["stop_sum"] / n_seq, 1),
                "mean_emissions": round(pol_stats[p]["emit_sum"] / n_seq, 2),
                "emission_rate_pct": round((pol_stats[p]["emit_count"] / n_seq) * 100.0, 1),
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ablation_study_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # write csv
    import csv
    with open(RESULTS_DIR / "ablation_study_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    generate_ablation_plots(summary, FIGURES_DIR)

    # print console overview
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS (N=14 Validation Trays)")
    print("=" * 80)
    print(f"{'Configuration':<42s} | {'Policy':<4s} | {'Acc (%)':<8s} | {'Avg Stop':<8s} | {'Avg Emit':<8s}")
    print("-" * 80)
    for cfg_name, data in summary.items():
        for i, p in enumerate(["M0", "M1", "M2", "M3", "MT"]):
            cfg_label = cfg_name if i == 0 else ""
            stats = data["policies"][p]
            print(
                f"{cfg_label:<42s} | {p:<4s} | "
                f"{stats['accuracy_pct']:>6.1f}% | "
                f"{stats['mean_stopping_frame']:>8.1f} | "
                f"{stats['mean_emissions']:>8.2f}"
            )
        print("-" * 80)
    print(f"\nsaved summary json : {RESULTS_DIR / 'ablation_study_summary.json'}")
    print(f"saved summary csv  : {RESULTS_DIR / 'ablation_study_results.csv'}")
    print(f"saved plots        : {FIGURES_DIR / 'val_ablation_accuracy_comparison.png'}")
    print(f"                     {FIGURES_DIR / 'val_ablation_stopping_frames.png'}")
    print("done.")


if __name__ == "__main__":
    main()
