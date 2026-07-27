"""
real_experiment.py - run the bayesian decision pipeline on real classifier trays.

loads exported (T, K) probability trays, runs all five policies unchanged,
captures belief trajectories + bocpd signals + confidence + stopping,
generates publication-quality plots matching the synthetic style,
and writes a structured json summary.

usage:
    cd src
    py -3.11 real_experiment.py                          # default tray
    py -3.11 real_experiment.py --tray ../my_tray.npy    # custom tray
    py -3.11 real_experiment.py --trays ../trays/         # directory of .npy

changes nothing in the existing codebase. K is inferred from each tray.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from belief import FlatBel
from changepoint import BOCPD
from gates import Cross
from metrics import emission_metrics
from policies import BayesThresh, BocpdThresh, RawThresh
from style import ACC, COL, DARK, GRID, BG, clean, save, use
from tree_belief import TreeBel, build_tree, specificity

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
DEFAULT_TRAY = ROOT / "data" / "sample_probability_tray.npy" if (ROOT / "data" / "sample_probability_tray.npy").exists() else ROOT / "sample_probability_tray.npy"

# class names placeholder - override via --labels if needed
CLASS_NAMES = None


def load_tray(path):
    p = np.load(path).astype(np.float64)
    if p.ndim != 2:
        raise ValueError(f"expected 2d tray, got {p.shape}")
    p = np.clip(p, 0.0, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def make_policies(K, tau=0.5):
    # identity sim - no confusion data yet
    S = np.eye(K)
    lo = max(0.0, tau - 0.15)
    return {
        "M0": RawThresh(K, theta=tau, low=lo),
        "M1": BayesThresh(K, lam=0.5, rho=0.0, tau=tau, low=lo),
        "M2": BayesThresh(K, lam=0.5, rho=0.05, tau=tau, low=lo),
        "M3": BocpdThresh(K, lam=0.5, tau=tau, cp_thr=0.1, low=lo),
        "MT": TreeBel(K, S, lam=0.5, rho=0.05, tau=tau, min_sp=0.25, low=lo),
    }, S


def run_diagnostics(P, K):
    T = len(P)
    frames = np.arange(T)
    preds = P.argmax(axis=1)
    maxconf = P.max(axis=1)

    # belief trajectories for static + switching
    fb_static = FlatBel(K, lam=0.5, rho=0.0)
    fb_switch = FlatBel(K, lam=0.5, rho=0.05)
    B_static = fb_static.run(P)
    B_switch = fb_switch.run(P)

    # bocpd
    cpd = BOCPD(K, hazard=1/30.0, alpha=1.0)
    cpd.reset()
    cp_posts = np.zeros(T)
    run_posteriors = np.full((T, T + 1), np.nan)
    for t in range(T):
        snap = cpd.update(int(preds[t]))
        cp_posts[t] = snap.cp_posterior
        run_posteriors[t, :len(snap.posterior)] = snap.posterior

    # run all policies, capture internals
    pols, S = make_policies(K)
    emissions = {}
    scores_by_pol = {}
    for name, pol in pols.items():
        em = pol.run(P)
        emissions[name] = em
        scores_by_pol[name] = np.array(pol.sc) if hasattr(pol, "sc") else maxconf.copy()

    # m3 changepoint internals
    m3 = pols["M3"]
    m3_cp = np.array(m3.cp) if hasattr(m3, "cp") else cp_posts
    m3_bd = list(m3.bd) if hasattr(m3, "bd") else []

    return {
        "T": T, "K": K, "frames": frames, "preds": preds, "maxconf": maxconf,
        "B_static": B_static, "B_switch": B_switch,
        "cp_posts": cp_posts, "run_posteriors": run_posteriors,
        "emissions": emissions, "scores": scores_by_pol,
        "m3_cp": m3_cp, "m3_bd": m3_bd, "S": S,
    }


def class_label(k):
    if CLASS_NAMES and k < len(CLASS_NAMES):
        return CLASS_NAMES[k]
    return f"class {k}"


# ── plot 1: belief evolution ──
def plot_belief_evolution(diag, P, prefix="real_"):
    use()
    T, K = diag["T"], diag["K"]
    frames = diag["frames"]
    B_static = diag["B_static"]
    B_switch = diag["B_switch"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")

    # top-left: raw posterior heatmap
    sns.heatmap(P.T, ax=ax[0, 0], cmap="viridis", vmin=0, vmax=P.max(),
                cbar_kws={"label": "Probability"})
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_title("Frame-level classifier posteriors")
    ax[0, 0].set_xlabel("Frame")
    ax[0, 0].set_ylabel("Class")

    # top-right: static belief heatmap
    sns.heatmap(B_static.T, ax=ax[0, 1], cmap="magma", vmin=0, vmax=1.0,
                cbar_kws={"label": "Belief"})
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title("Static Bayesian belief ($\\rho=0$)")
    ax[0, 1].set_xlabel("Frame")
    ax[0, 1].set_ylabel("Class")

    # bottom-left: bmax trajectories
    sns.lineplot(x=frames, y=B_static.max(axis=1), ax=ax[1, 0],
                 label="Static $b_{\\max}$", linewidth=2.0, color=COL["M1"])
    sns.lineplot(x=frames, y=B_switch.max(axis=1), ax=ax[1, 0],
                 label="Switching $b_{\\max}$", linewidth=2.0, color=COL["M2"])
    sns.lineplot(x=frames, y=diag["maxconf"], ax=ax[1, 0],
                 label="Raw $\\max p$", linewidth=1.2, alpha=0.6, color=COL["M0"])
    ax[1, 0].set_ylim(0, 1.02)
    ax[1, 0].set_title("Belief confidence trajectories")
    ax[1, 0].set_xlabel("Frame")
    ax[1, 0].set_ylabel("Confidence")
    ax[1, 0].legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    clean(ax[1, 0])

    # bottom-right: switching belief heatmap
    sns.heatmap(B_switch.T, ax=ax[1, 1], cmap="magma", vmin=0, vmax=1.0,
                cbar_kws={"label": "Belief"})
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title("Switching Bayesian belief ($\\rho=0.05$)")
    ax[1, 1].set_xlabel("Frame")
    ax[1, 1].set_ylabel("Class")

    fig.suptitle("Bayesian belief evolution - real classifier",
                 fontsize=15, fontweight="bold")
    save(fig, f"{prefix}belief_evolution.png")


# ── plot 2: stopping policy visualization ──
def plot_stopping(diag, P, prefix="real_"):
    use()
    T = diag["T"]
    frames = diag["frames"]
    emissions = diag["emissions"]
    scores = diag["scores"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), layout="constrained",
                           height_ratios=[2, 1])

    # top: confidence scores per policy with emission markers
    for name in ["M0", "M1", "M2", "M3"]:
        sc = scores[name]
        sns.lineplot(x=frames[:len(sc)], y=sc, ax=ax[0],
                     label=name, color=COL[name], linewidth=1.6, alpha=0.85)
        em = emissions[name]
        if em:
            ex = [t for t, _ in em]
            ey = [sc[t] for t, _ in em]
            ax[0].scatter(ex, ey, color=COL[name], marker="v", s=80,
                          zorder=5, edgecolors=DARK, linewidths=0.8)

    ax[0].axhline(0.5, color="#7A8793", linestyle=":", linewidth=1.0,
                  label="$\\tau=0.5$", alpha=0.7)
    ax[0].set_ylim(0, 1.02)
    ax[0].set_title("Decision scores and emission events")
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("Score")
    ax[0].legend(ncol=5, loc="upper right", frameon=True,
                 facecolor="white", framealpha=0.95, fontsize=9)
    clean(ax[0])

    # bottom: emission timeline
    names = ["M0", "M1", "M2", "M3", "MT"]
    tx_list, k_list = [], []
    for name in names:
        for t, _ in emissions[name]:
            tx_list.append(t)
            k_list.append(name)

    if tx_list:
        sns.scatterplot(x=tx_list, y=k_list, hue=k_list,
                        palette={n: COL[n] for n in names},
                        s=90, ax=ax[1], legend=False,
                        edgecolor=DARK, linewidth=0.8)
    ax[1].set_xlim(-0.5, T - 0.5)
    ax[1].set_title("Emission timeline")
    ax[1].set_xlabel("Frame")
    clean(ax[1])

    fig.suptitle("Stopping policy visualization - real classifier",
                 fontsize=15, fontweight="bold")
    save(fig, f"{prefix}stopping_policy.png")


# ── plot 3: confidence vs frame ──
def plot_confidence(diag, P, prefix="real_"):
    use()
    T, K = diag["T"], diag["K"]
    frames = diag["frames"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), layout="constrained",
                           height_ratios=[2, 1])

    # top: per-class raw posterior
    for k in range(K):
        ax[0].plot(frames, P[:, k], linewidth=1.4, alpha=0.8,
                   label=class_label(k))
    ax[0].set_ylim(0, min(1.0, P.max() * 1.15))
    ax[0].set_title("Per-class posterior probability")
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("$p(k | x_t)$")
    ax[0].legend(ncol=min(K, 5), loc="upper right", frameon=True,
                 facecolor="white", framealpha=0.95, fontsize=9)
    clean(ax[0])

    # bottom: entropy
    ent = -np.sum(P * np.log(np.clip(P, 1e-12, None)), axis=1)
    norm_ent = ent / np.log(K)
    sns.lineplot(x=frames, y=norm_ent, ax=ax[1], color=ACC, linewidth=2.0)
    ax[1].set_ylim(0, 1.02)
    ax[1].set_title("Normalized entropy")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("$H / \\log K$")
    clean(ax[1])

    fig.suptitle("Confidence evolution - real classifier",
                 fontsize=15, fontweight="bold")
    save(fig, f"{prefix}confidence_evolution.png")


# ── plot 4: predicted class timeline ──
def plot_class_timeline(diag, P, prefix="real_"):
    use()
    T, K = diag["T"], diag["K"]
    frames = diag["frames"]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), layout="constrained",
                           height_ratios=[1, 1, 1])

    # top: raw argmax
    preds = diag["preds"]
    ax[0].step(frames, preds, where="mid", color=COL["M0"], linewidth=2.0)
    ax[0].set_yticks(range(K))
    ax[0].set_yticklabels([class_label(k) for k in range(K)])
    ax[0].set_title("Raw classifier prediction (argmax)")
    ax[0].set_xlabel("Frame")
    clean(ax[0])

    # middle: static belief argmax
    b_pred = diag["B_static"].argmax(axis=1)
    ax[1].step(frames, b_pred, where="mid", color=COL["M1"], linewidth=2.0)
    ax[1].set_yticks(range(K))
    ax[1].set_yticklabels([class_label(k) for k in range(K)])
    ax[1].set_title("Static belief prediction ($\\rho=0$)")
    ax[1].set_xlabel("Frame")
    clean(ax[1])

    # bottom: switching belief argmax
    sw_pred = diag["B_switch"].argmax(axis=1)
    ax[2].step(frames, sw_pred, where="mid", color=COL["M2"], linewidth=2.0)
    ax[2].set_yticks(range(K))
    ax[2].set_yticklabels([class_label(k) for k in range(K)])
    ax[2].set_title("Switching belief prediction ($\\rho=0.05$)")
    ax[2].set_xlabel("Frame")
    clean(ax[2])

    fig.suptitle("Predicted class timeline - real classifier",
                 fontsize=15, fontweight="bold")
    save(fig, f"{prefix}class_timeline.png")


# ── plot 5: changepoint visualization ──
def plot_changepoint(diag, P, prefix="real_"):
    use()
    T = diag["T"]
    frames = diag["frames"]

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), layout="constrained",
                           height_ratios=[1, 2])

    # top: cp posterior
    sns.lineplot(x=frames, y=diag["m3_cp"], ax=ax[0],
                 color=COL["M3"], linewidth=2.0, label="CP posterior")
    ax[0].axhline(0.1, color="#E7298A", linestyle="--", linewidth=1.0,
                  alpha=0.7, label="$\\theta_{cp}=0.10$")
    ax[0].axhline(1/30.0, color="#7A8793", linestyle=":", linewidth=1.0,
                  alpha=0.6, label="Prior hazard")
    if diag["m3_bd"]:
        bd_y = [diag["m3_cp"][t] for t in diag["m3_bd"]]
        ax[0].scatter(diag["m3_bd"], bd_y, color="#E7298A", marker="x",
                      s=100, zorder=5, linewidths=2.5, label="Detected")
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("BOCPD changepoint posterior")
    ax[0].set_xlabel("Frame")
    ax[0].set_ylabel("$P(r_t = 0)$")
    ax[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    clean(ax[0])

    # bottom: run-length posterior heatmap
    rp = diag["run_posteriors"]
    max_rl = min(T, 50)
    sns.heatmap(rp[:, :max_rl].T, ax=ax[1], cmap="magma",
                vmin=0, vmax=np.nanmax(rp),
                cbar_kws={"label": "Posterior mass"})
    ax[1].invert_yaxis()
    ax[1].set_title("Run-length posterior")
    ax[1].set_xlabel("Frame")
    ax[1].set_ylabel("Run length")
    if diag["m3_bd"]:
        for bd in diag["m3_bd"]:
            ax[1].axvline(bd, color="cyan", linestyle="--", linewidth=1.0, alpha=0.8)

    fig.suptitle("Changepoint detection - real classifier",
                 fontsize=15, fontweight="bold")
    save(fig, f"{prefix}changepoint.png")


# ── plot 6: experiment summary ──
def plot_summary(diag, P, tray_path, prefix="real_"):
    use()
    T, K = diag["T"], diag["K"]
    emissions = diag["emissions"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")

    # top-left: emission count bar chart
    names = ["M0", "M1", "M2", "M3", "MT"]
    counts = [len(emissions[n]) for n in names]
    colors = [COL[n] for n in names]
    sns.barplot(x=names, y=counts, hue=names, palette=colors,
                legend=False, ax=ax[0, 0], width=0.72,
                alpha=0.9, edgecolor=DARK, linewidth=0.8)
    ax[0, 0].set_title("Emissions per policy")
    ax[0, 0].set_ylabel("Count")
    clean(ax[0, 0])

    # top-right: final belief distribution (switching)
    final_b = diag["B_switch"][-1]
    x_classes = [class_label(k) for k in range(K)]
    bar_colors = sns.color_palette("viridis", K)
    sns.barplot(x=x_classes, y=final_b, hue=x_classes, palette=bar_colors,
                legend=False, ax=ax[0, 1], width=0.72,
                alpha=0.9, edgecolor=DARK, linewidth=0.8)
    ax[0, 1].set_title("Final switching belief distribution")
    ax[0, 1].set_ylabel("$b_T(k)$")
    ax[0, 1].set_ylim(0, 1.02)
    clean(ax[0, 1])

    # bottom-left: bmax + emission overlay
    frames = diag["frames"]
    bmax = diag["B_switch"].max(axis=1)
    sns.lineplot(x=frames, y=bmax, ax=ax[1, 0], color=COL["M2"],
                 linewidth=2.0, label="Switching $b_{\\max}$")
    ax[1, 0].axhline(0.5, color="#7A8793", linestyle=":", alpha=0.6)
    for name in ["M2"]:
        em = emissions[name]
        if em:
            ex = [t for t, _ in em]
            ey = [bmax[t] for t, _ in em]
            ax[1, 0].scatter(ex, ey, color=COL[name], marker="v",
                             s=80, zorder=5, edgecolors=DARK, linewidths=0.8,
                             label=f"{name} emissions")
    ax[1, 0].set_ylim(0, 1.02)
    ax[1, 0].set_title("Switching belief with emissions")
    ax[1, 0].set_xlabel("Frame")
    ax[1, 0].set_ylabel("$b_{\\max}$")
    ax[1, 0].legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    clean(ax[1, 0])

    # bottom-right: text summary
    ax[1, 1].axis("off")
    majority = int(np.bincount(P.argmax(axis=1), minlength=K).argmax())
    final_pred = int(diag["B_switch"].argmax(axis=1)[-1])
    final_bmax = float(bmax[-1])
    cp_count = len(diag["m3_bd"])

    lines = [
        f"Tray: {Path(tray_path).name}",
        f"Frames: {T}    Classes: {K}",
        f"Raw majority: {class_label(majority)}",
        f"Final belief: {class_label(final_pred)}  ($b_{{\\max}}$={final_bmax:.3f})",
        f"Changepoints detected: {cp_count}",
        "",
    ]
    for name in names:
        em = emissions[name]
        ids = []
        for _, y in em:
            if isinstance(y, (set, frozenset)):
                ids.append("{" + ",".join(str(v) for v in sorted(y)) + "}")
            elif y is not None:
                ids.append(str(y))
        lines.append(f"{name}: {len(em)} emission(s)  {ids}")

    ax[1, 1].text(0.05, 0.95, "\n".join(lines), transform=ax[1, 1].transAxes,
                  fontsize=11, verticalalignment="top", fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                            edgecolor=GRID, alpha=0.95))

    fig.suptitle("Experiment summary - real classifier",
                 fontsize=15, fontweight="bold")
    save(fig, f"{prefix}summary.png")


def build_report(diag, P, tray_path):
    T, K = diag["T"], diag["K"]
    emissions = diag["emissions"]
    majority = int(np.bincount(P.argmax(axis=1), minlength=K).argmax())

    report = {
        "tray": str(tray_path),
        "T": T,
        "K": K,
        "raw_majority_class": majority,
        "final_static_prediction": int(diag["B_static"].argmax(axis=1)[-1]),
        "final_switching_prediction": int(diag["B_switch"].argmax(axis=1)[-1]),
        "final_static_bmax": float(diag["B_static"].max(axis=1)[-1]),
        "final_switching_bmax": float(diag["B_switch"].max(axis=1)[-1]),
        "raw_confidence_range": [float(diag["maxconf"].min()), float(diag["maxconf"].max())],
        "bocpd_detections": diag["m3_bd"],
        "policies": {},
    }

    for name in ["M0", "M1", "M2", "M3", "MT"]:
        em = emissions[name]
        em_list = []
        for t, y in em:
            if isinstance(y, (set, frozenset)):
                em_list.append({"frame": t, "identity": sorted(y)})
            elif y is not None:
                em_list.append({"frame": t, "identity": int(y)})
        report["policies"][name] = {
            "n_emissions": len(em),
            "emissions": em_list,
        }

    return report


def run_one(tray_path, prefix="real_"):
    P = load_tray(tray_path)
    T, K = P.shape
    print(f"\n{'='*60}")
    print(f"tray: {tray_path}")
    print(f"shape: T={T}, K={K}")

    diag = run_diagnostics(P, K)

    # console summary
    print(f"raw confidence: [{diag['maxconf'].min():.3f}, {diag['maxconf'].max():.3f}]")
    print(f"static bmax final: {diag['B_static'].max(axis=1)[-1]:.4f} -> class {diag['B_static'].argmax(axis=1)[-1]}")
    print(f"switching bmax final: {diag['B_switch'].max(axis=1)[-1]:.4f} -> class {diag['B_switch'].argmax(axis=1)[-1]}")
    if diag["m3_bd"]:
        print(f"bocpd changepoints at frames: {diag['m3_bd']}")
    else:
        print("bocpd: no changepoints detected")

    print(f"\n{'policy':>6s}  {'n_emit':>6s}  {'emissions':>40s}")
    print("-" * 56)
    for name in ["M0", "M1", "M2", "M3", "MT"]:
        em = diag["emissions"][name]
        em_str = [(t, str(y) if not isinstance(y, (set, frozenset))
                   else "{" + ",".join(str(v) for v in sorted(y)) + "}")
                  for t, y in em]
        print(f"{name:>6s}  {len(em):>6d}  {str(em_str):>40s}")

    # generate all 6 plots
    print(f"\ngenerating plots with prefix '{prefix}'...")
    plot_belief_evolution(diag, P, prefix)
    plot_stopping(diag, P, prefix)
    plot_confidence(diag, P, prefix)
    plot_class_timeline(diag, P, prefix)
    plot_changepoint(diag, P, prefix)
    plot_summary(diag, P, tray_path, prefix)

    # save json report
    report = build_report(diag, P, tray_path)
    RES.mkdir(exist_ok=True)
    out = RES / f"{prefix}report.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(f"\nreport: {out}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="real classifier tray -> bayesian pipeline + plots")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tray", type=str, default=None,
                       help="path to a single .npy probability tray")
    group.add_argument("--trays", type=str, default=None,
                       help="directory of .npy probability trays")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="class names (e.g. --labels hello thanks sorry yes no)")
    args = parser.parse_args()

    global CLASS_NAMES
    if args.labels:
        CLASS_NAMES = args.labels

    if args.trays:
        tray_dir = Path(args.trays)
        trays = sorted(tray_dir.glob("*.npy"))
        if not trays:
            raise FileNotFoundError(f"no .npy files in {tray_dir}")
        for i, path in enumerate(trays):
            prefix = f"real_{path.stem}_"
            run_one(path, prefix)
    else:
        tray_path = args.tray or str(DEFAULT_TRAY)
        run_one(tray_path, prefix="real_")

    print("\ndone.")


if __name__ == "__main__":
    main()
