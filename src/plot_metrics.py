from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evidence import Simcon, concat_stream
from metrics import emission_metrics, pool
from policies import BayesThresh, BocpdThresh, RawThresh
from tree_belief import (
    TreeBel,
    confusion_to_sim,
    estimate_confusion,
)


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def val(x):
    return 0.0 if not np.isfinite(x) else x


def put_na(ax, xs, ys):
    for x, y in zip(xs, ys):
        if not np.isfinite(y):
            ax.text(
                x,
                0.02,
                "NA",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )


def main():
    import seaborn as sns
    cfg = Simcon()
    rng = np.random.default_rng(51)

    Ps, Zs = [], []

    for _ in range(60):
        s = concat_stream(cfg, rng, n_signs=4)
        Ps.append(s["p"])
        Zs.append(s["z"])

    C = estimate_confusion(Ps, Zs, cfg.K)
    S = confusion_to_sim(C)

    ps = {
        "M0": RawThresh(cfg.K, theta=0.8),
        "M1": BayesThresh(
            cfg.K, lam=0.5, rho=0.0, tau=0.8
        ),
        "M2": BayesThresh(
            cfg.K, lam=0.5, rho=0.05, tau=0.8
        ),
        "M3": BocpdThresh(
            cfg.K,
            lam=0.5,
            tau=0.8,
            cp_thr=0.1,
        ),
        "MT": TreeBel(
            cfg.K,
            S,
            lam=0.5,
            rho=0.05,
            tau=0.8,
            min_sp=0.25,
        ),
    }

    rows = {k: [] for k in ps}

    for _ in range(100):
        s = concat_stream(cfg, rng, n_signs=5)

        for k, p in ps.items():
            em = p.run(s["p"])

            r = emission_metrics(
                em,
                s["z"],
                s["labels"],
                s["onsets"],
                K=cfg.K,
            )

            rows[k].append(r)

    res = {
        k: pool(v)
        for k, v in rows.items()
    }

    names = list(res)
    x = np.arange(len(names))

    fig, ax = plt.subplots(
        3,
        2,
        figsize=(13, 12),
        layout="constrained",
    )

    x1, y1, h1 = [], [], []
    for key in ["rep_r", "per_r", "err_r"]:
        for k in names:
            x1.append(k)
            y1.append(val(res[k][key]))
            h1.append(key)
    
    sns.barplot(x=x1, y=y1, hue=h1, ax=ax[0, 0])
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("Output error rates")
    ax[0, 0].legend()

    cov = np.array([res[k]["cov"] for k in names])

    sns.barplot(x=names, y=[val(v) for v in cov], ax=ax[0, 1])
    put_na(ax[0, 1], x, cov)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_title("Boundary coverage")
    ax[0, 1].set_ylabel("Fraction hit")

    rd = np.array([res[k]["rdel"] for k in names])
    hd = np.array([res[k]["hdel"] for k in names])

    x2, y2, h2 = [], [], []
    for nm in names:
        x2.extend([nm, nm])
        y2.extend([val(res[nm]["rdel"]), val(res[nm]["hdel"])])
        h2.extend(["restricted delay", "hit-only delay"])

    sns.barplot(x=x2, y=y2, hue=h2, ax=ax[1, 0])
    ax[1, 0].set_title("Boundary delay")
    ax[1, 0].set_ylabel("Frames")
    ax[1, 0].legend()

    hc = np.array([res[k]["hcov_r"] for k in names])
    sp = np.array([res[k]["spec"] for k in names])

    x3, y3, h3 = [], [], []
    for nm in names:
        x3.extend([nm, nm])
        y3.extend([val(res[nm]["hcov_r"]), val(res[nm]["spec"])])
        h3.extend(["coverage", "specificity"])

    sns.barplot(x=x3, y=y3, hue=h3, ax=ax[1, 1])
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_title("Coverage versus specificity")
    ax[1, 1].legend()

    sq = np.array([res[k]["seq_r"] for k in names])

    sns.barplot(x=names, y=[val(v) for v in sq], ax=ax[2, 0])
    put_na(ax[2, 0], x, sq)
    ax[2, 0].set_title("Normalized sequence edit distance")
    ax[2, 0].set_ylabel("Lower is better")

    ne = np.array([
        res[k]["n_emit"] / 100
        for k in names
    ])

    sns.barplot(x=names, y=ne, ax=ax[2, 1])
    ax[2, 1].axhline(
        5,
        color="red",
        linestyle="--",
        label="true signs per stream",
    )
    ax[2, 1].set_title("Mean emissions per stream")
    ax[2, 1].set_ylabel("Count")
    ax[2, 1].legend()

    fig.suptitle(
        "Metric diagnostics — 100 held-out streams",
        fontsize=15,
        fontweight="bold",
    )

    FIG.mkdir(exist_ok=True)
    out = FIG / "metric_diagnostics.png"

    fig.savefig(
        out,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(
        f"{'id':>3} | {'emit':>5} | {'rep':>6} | "
        f"{'per':>6} | {'err':>6} | {'cov':>6} | "
        f"{'rdel':>6} | {'hcov':>6} | {'spec':>6}"
    )

    for k in names:
        r = res[k]

        def f(v):
            return "    NA" if not np.isfinite(v) else f"{v:6.3f}"

        print(
            f"{k:>3} | "
            f"{r['n_emit'] / 100:5.2f} | "
            f"{f(r['rep_r'])} | "
            f"{f(r['per_r'])} | "
            f"{f(r['err_r'])} | "
            f"{f(r['cov'])} | "
            f"{f(r['rdel'])} | "
            f"{f(r['hcov_r'])} | "
            f"{f(r['spec'])}"
        )

    print("saved:", out)


if __name__ == "__main__":
    main()
