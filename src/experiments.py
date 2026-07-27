import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from belief import FlatBel
from evidence import Simcon, generate_stream
from metrics import COUNT_FIELDS, emission_metrics, pool
from policies import BayesThresh, BocpdThresh, RawThresh
from protocol import SCENARIOS, content_hash
from reporting import (
    METHODS,
    PUBLICATION_ARTIFACTS,
    ReportValidationError,
    build_headline_report,
    build_reduced_report,
    load_report_json,
    write_report_json,
)
from stopping import (
    COMPARATOR_LABEL,
    calibration,
    evaluate_stopping,
    fit_empirical_dp,
    freeze_stopping_policies,
    select_fixed_threshold,
)
from style import COL, DARK, RULE, clean, save, use
from tree_belief import (
    TreeBel,
    confusion_to_sim,
    estimate_confusion,
)


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

METS = [
    "emit",
    "rep_r",
    "per_r",
    "err_r",
    "cov",
    "rdel",
    "hdel",
    "hcov_r",
    "spec",
    "seq_r",
]


def mk_single(cfg, rng, n, lam=0.5, *, role="fit", seed=0):
    fb = FlatBel(cfg.K, lam=lam, rho=0.0)
    out = []

    for index in range(n):
        stream = generate_stream(
            cfg,
            rng,
            role=role,
            scenario=SCENARIOS[index % len(SCENARIOS)],
            seed=seed,
            within_seed_index=index,
            n_signs=1,
        )
        B = fb.run(stream.policy)

        out.append({
            "stream_id": stream.manifest.stream_id,
            "role": role,
            "manifest": stream.manifest,
            "policy": stream.policy,
            "scoring": stream.scoring,
            "bmax": B.max(axis=1),
            "pred": B.argmax(axis=1),
            "truth": int(stream.scoring.truth[0]),
        })

    return out


def mk_cont(cfg, rng, n, ns=5, *, role="test", seed=0):
    return [
        generate_stream(
            cfg,
            rng,
            role=role,
            scenario=SCENARIOS[index % len(SCENARIOS)],
            seed=seed,
            within_seed_index=index,
            n_signs=ns,
        )
        for index in range(n)
    ]


def mk_sim(ss, K):
    Ps = [stream.policy.posteriors for stream in ss]
    Zs = [stream.scoring.truth for stream in ss]

    C = estimate_confusion(Ps, Zs, K)
    return confusion_to_sim(C)


def mk_pols(cfg, S):
    return {
        "M0": RawThresh(
            cfg.K,
            theta=0.8,
        ),
        "M1": BayesThresh(
            cfg.K,
            lam=0.5,
            rho=0.0,
            tau=0.8,
        ),
        "M2": BayesThresh(
            cfg.K,
            lam=0.5,
            rho=0.05,
            tau=0.8,
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


def score_one(pol, ss, cfg):
    rows = []

    for stream in ss:
        # The policy receives no truth, family, onset, bounds, or settings.
        em = pol.run(stream.policy)
        scoring = stream.scoring

        rows.append(emission_metrics(
            em,
            np.asarray(scoring.truth, dtype=int),
            scoring.labels,
            scoring.onsets,
            K=cfg.K,
        ))

    pooled = pool(rows)
    result = pooled.to_dict(include_details=True)
    result["emit"] = pooled["n_emit"] / len(ss) if ss else float("nan")

    return result


def score_all(pols, ss, cfg):
    return {
        k: score_one(p, ss, cfg)
        for k, p in pols.items()
    }


def boot(xs, nb=10000, seed=0):
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]

    if len(a) == 0:
        return {
            "mean": None,
            "lo": None,
            "hi": None,
            "n": 0,
        }

    m = float(a.mean())

    if len(a) == 1:
        return {
            "mean": m,
            "lo": m,
            "hi": m,
            "n": 1,
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0,
        len(a),
        size=(nb, len(a)),
    )
    bs = a[idx].mean(axis=1)

    return {
        "mean": m,
        "lo": float(np.quantile(bs, 0.025)),
        "hi": float(np.quantile(bs, 0.975)),
        "n": len(a),
    }


def one(seed, cfg, n):
    ss = np.random.SeedSequence(seed)
    sf, sd, sx, st = ss.spawn(4)

    rf = np.random.default_rng(sf)
    rd = np.random.default_rng(sd)
    rx = np.random.default_rng(sx)
    rt = np.random.default_rng(st)

    # Fit estimates tables and development selects scalar choices. Every
    # ratio is frozen before the held-out test streams are generated/accessed.
    fit = mk_single(cfg, rf, n["sf"], role="fit", seed=seed)
    dev = mk_single(cfg, rd, n["sd"], role="development", seed=seed)
    ratios = [0.02, 0.05, 0.10, 0.20]
    frozen = []

    for cw in ratios:
        stop_config = {
            "wait_cost": cw,
            "error_cost": 1.0,
            "bin_count": 25,
            "empty_bin_fallback": {
                "accuracy": "bin_center",
                "transition": "self",
            },
            # Sparse empirical states use the declared fallback; supported
            # bins receive mild shrinkage toward that same frozen fallback.
            "minimum_bin_visits": 5,
            "minimum_transition_visits": 3,
            "accuracy_prior_strength": 2.0,
        }
        dp_policy = fit_empirical_dp(fit, stop_config)
        fixed_policy = select_fixed_threshold(dev, {
            **stop_config,
            "thresholds": np.linspace(0.30, 0.99, 50).tolist(),
        })
        frozen.append(freeze_stopping_policies(
            stop_config, dp_policy, fixed_policy
        ))

    test = mk_single(cfg, rt, n["st"], role="test", seed=seed)
    front = []
    for cw, policies in zip(ratios, frozen):
        evaluation = evaluate_stopping(test, policies)
        methods = evaluation["methods"]
        dp_policy = policies["policies"]["empirical_dp"]
        fixed_policy = policies["policies"]["fixed"]
        front.append({
            "ratio": cw,
            "tau": policies["policies"]["myopic"]["threshold"],
            "fixed_tau": fixed_policy["threshold"],
            "immediate": methods["immediate"],
            "myopic": methods["myopic"],
            "fixed": methods["fixed"],
            "dp": methods["empirical_dp"],
            "dp_label": COMPARATOR_LABEL,
            "paired_differences": evaluation["paired_differences"],
            "ordered_test_stream_ids": evaluation["ordered_test_stream_ids"],
            "dp_diagnostics": dp_policy["diagnostics"],
            "frozen_stopping_config": dp_policy["config"],
        })

    ctr, acc, tot = calibration(test, nb=15)
    hit = np.where(tot > 0, acc * tot, 0.0)

    # Continuous roles. Taxonomy/confusion uses fit only. Frozen policies are
    # scored on one paired test list; stress streams are mechanism checks only.
    cfit = mk_cont(cfg, rf, n["cf"], role="fit", seed=seed)
    cstress = mk_cont(cfg, rx, n["ct"], role="stress", seed=seed)
    ctest = mk_cont(cfg, rt, n["ct"], role="test", seed=seed)

    S = mk_sim(cfit, cfg.K)
    policies = mk_pols(cfg, S)
    cont = score_all(policies, ctest, cfg)
    scenario_cont = {}
    scenario_test_ids = {}
    for scenario in SCENARIOS:
        scenario_streams = [
            stream for stream in ctest if stream.manifest.scenario == scenario
        ]
        scenario_test_ids[scenario] = tuple(
            stream.manifest.stream_id for stream in scenario_streams
        )
        scenario_cont[scenario] = score_all(
            mk_pols(cfg, S), scenario_streams, cfg
        )

    # Predeclared rho ablation. rho=0 is the static limit.
    rho_out = {}

    for rho in [0.0, 0.01, 0.03, 0.05, 0.08, 0.12]:
        p = BayesThresh(
            cfg.K,
            lam=0.5,
            rho=rho,
            tau=0.8,
        )
        rho_out[str(rho)] = score_one(p, cstress, cfg)

    # BOCPD threshold sensitivity is a predeclared frozen mechanism check.
    cp_out = {}

    for cp in [0.06, 0.08, 0.10, 0.12, 0.14]:
        p = BocpdThresh(
            cfg.K,
            lam=0.5,
            tau=0.8,
            cp_thr=cp,
        )
        cp_out[str(cp)] = score_one(p, cstress, cfg)

    # Tree specificity sensitivity is checked only on stress streams.
    mt_out = {}

    for ms in [0.25, 0.40, 0.60, 0.80]:
        p = TreeBel(
            cfg.K,
            S,
            lam=0.5,
            rho=0.05,
            tau=0.8,
            min_sp=ms,
        )
        mt_out[str(ms)] = score_one(p, cstress, cfg)

    return {
        "seed": seed,
        "front": front,
        "rel": {
            "ctr": ctr,
            "hit": hit,
            "tot": tot,
        },
        "cont": cont,
        "scenario_cont": scenario_cont,
        "scenario_test_ids": scenario_test_ids,
        "stopping_test_ids": tuple(row["stream_id"] for row in test),
        "role_stream_ids": {
            "fit": tuple(
                row["stream_id"] for row in fit
            ) + tuple(stream.manifest.stream_id for stream in cfit),
            "development": tuple(row["stream_id"] for row in dev),
            "stress": tuple(stream.manifest.stream_id for stream in cstress),
            "test": tuple(row["stream_id"] for row in test) + tuple(
                stream.manifest.stream_id for stream in ctest
            ),
        },
        "rho": rho_out,
        "cp": cp_out,
        "mt": mt_out,
    }


def sum_block(raw, name, vals, nb, off):
    out = {}

    for x in vals:
        key = str(x)
        out[key] = {}

        for met in METS:
            xs = [
                r[name][key][met]
                for r in raw
            ]
            out[key][met] = boot(
                xs,
                nb=nb,
                seed=off,
            )
            off += 1

    return out, off


def summarize(raw, nb):
    out = {
        "cont": {},
        "front": {},
    }
    off = 1000

    for m in ["M0", "M1", "M2", "M3", "MT"]:
        out["cont"][m] = {}

        for met in METS:
            xs = [r["cont"][m][met] for r in raw]
            out["cont"][m][met] = boot(
                xs,
                nb=nb,
                seed=off,
            )
            off += 1

    for i, ratio in enumerate([0.02, 0.05, 0.10, 0.20]):
        rk = str(ratio)
        out["front"][rk] = {}

        for rule in ["immediate", "myopic", "fixed", "dp"]:
            out["front"][rk][rule] = {}

            for met in ["risk", "acc", "latency"]:
                xs = [
                    r["front"][i][rule][met]
                    for r in raw
                ]
                out["front"][rk][rule][met] = boot(
                    xs,
                    nb=nb,
                    seed=off,
                )
                off += 1

        out["front"][rk]["paired_differences"] = {}
        for pair in ["dp_minus_myopic", "dp_minus_fixed"]:
            xs = [
                r["front"][i]["paired_differences"][pair]["risk"]
                for r in raw
            ]
            out["front"][rk]["paired_differences"][pair] = {
                "risk": boot(xs, nb=nb, seed=off),
                "units": "cost per stream",
            }
            off += 1

    out["rho"], off = sum_block(
        raw,
        "rho",
        [0.0, 0.01, 0.03, 0.05, 0.08, 0.12],
        nb,
        off,
    )

    out["cp"], off = sum_block(
        raw,
        "cp",
        [0.06, 0.08, 0.10, 0.12, 0.14],
        nb,
        off,
    )

    out["mt"], off = sum_block(
        raw,
        "mt",
        [0.25, 0.40, 0.60, 0.80],
        nb,
        off,
    )

    return out


def err(s):
    m = s["mean"]
    if m is None:
        return None, None
    lo = m - s["lo"]
    hi = s["hi"] - m
    return m, [[lo], [hi]]


def bars(ax, sm, met, title, ylim=None):
    import seaborn as sns
    names = list(sm)
    y_list, c_list = [], []
    err_dict = {}
    for i, k in enumerate(names):
        s = sm[k][met]
        y, ye = err(s)
        if y is None:
            ax.text(i, 0.04, "N/A", ha="center", va="bottom", fontsize=10, color=DARK)
            y_list.append(np.nan)
        else:
            y_list.append(y)
            err_dict[i] = ye
        c_list.append(COL[k])

    sns.barplot(
        x=names,
        y=y_list,
        hue=names,
        palette=c_list,
        legend=False,
        ax=ax,
        width=0.72,
        alpha=0.9,
        edgecolor=DARK,
        linewidth=0.8,
    )

    for i in err_dict:
        ax.errorbar(
            i,
            y_list[i],
            yerr=err_dict[i],
            color="#2F3E46",
            capsize=4,
            linewidth=1.1,
            fmt="none",
        )

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean(ax)


def plot_cont(sm, pre=""):
    use()
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(14, 13),
        layout="constrained",
    )
    bars(
        ax[0, 0],
        sm,
        "per_r",
        "Perseveration rate",
        (0, 0.45),
    )
    bars(
        ax[0, 1],
        sm,
        "cov",
        "Boundary coverage",
        (0, 1.05),
    )
    bars(
        ax[1, 0],
        sm,
        "rdel",
        "Failure-aware boundary delay",
    )
    ax[1, 0].set_ylabel("Frames")
    bars(
        ax[1, 1],
        sm,
        "spec",
        "Average specificity",
        (0, 1.05),
    )
    bars(
        ax[2, 0],
        sm,
        "seq_r",
        "Normalized edit distance",
    )
    ax[2, 0].set_ylabel("Error rate")
    bars(
        ax[2, 1],
        sm,
        "n_emit",
        "Total emissions per stream",
    )
    ax[2, 1].axhline(
        25,
        color="red",
        linestyle="--",
        label="True segments",
    )
    ax[2, 1].legend()
    save(fig, f"{pre}continuous.png")


def plot_front(sm, pre=""):
    use()
    fig, ax = plt.subplots(figsize=(5.5, 4.0), layout="constrained")

    for rule in ["immediate", "myopic", "fixed", "dp"]:
        xs, ys = [], []
        xlo, xhi = [], []
        ylo, yhi = [], []

        for rk, row in sm.items():
            a = row[rule]["latency"]
            b = row[rule]["acc"]

            xs.append(a["mean"])
            ys.append(b["mean"])

            xlo.append(a["mean"] - a["lo"])
            xhi.append(a["hi"] - a["mean"])
            ylo.append(b["mean"] - b["lo"])
            yhi.append(b["hi"] - b["mean"])

        ax.errorbar(
            xs,
            ys,
            xerr=[xlo, xhi],
            yerr=[ylo, yhi],
            marker="o",
            markersize=6,
            capsize=3.5,
            color=RULE[rule],
            label=rule,
            markeredgecolor=DARK,
            markeredgewidth=0.8,
            linewidth=1.8,
        )
        ax.fill_between(
            xs,
            [b - l for b, l in zip(ys, ylo)],
            [b + h for b, h in zip(ys, yhi)],
            color=RULE[rule],
            alpha=0.2,
        )

    off = {
        "0.02": (-55, 12),
        "0.05": (12, -10),
        "0.1": (12, -14),
        "0.2": (12, -18),
    }

    for rk, row in sm.items():
        x = row["myopic"]["latency"]["mean"]
        y = row["myopic"]["acc"]["mean"]

        ax.annotate(
            f"$c_w/c_e={rk}$",
            (x, y),
            xytext=off[rk],
            textcoords="offset points",
            fontsize=9.5,
            color=RULE["myopic"],
        )

    ax.margins(x=0.08, y=0.08)
    ax.set_xlabel("Commit latency (frames)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Single-sign stopping frontier")
    ax.locator_params(axis="x", nbins=6)
    ax.locator_params(axis="y", nbins=6)
    ax.legend(title="Rule", loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    clean(ax)
    save(fig, f"{pre}frontier.png")


def plot_rel(raw, pre=""):
    import seaborn as sns
    use()

    ctr = np.asarray(raw[0]["rel"]["ctr"])
    hit = sum(np.asarray(r["rel"]["hit"]) for r in raw)
    tot = sum(np.asarray(r["rel"]["tot"]) for r in raw)

    acc = np.divide(
        hit,
        tot,
        out=np.full_like(hit, np.nan, dtype=float),
        where=tot > 0,
    )

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(5.5, 5.0),
        layout="constrained",
        height_ratios=[2, 1],
    )

    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        ax=ax[0],
        linestyle="--",
        color="#7A8793",
        label="perfect calibration",
    )
    sns.lineplot(
        x=ctr,
        y=acc,
        ax=ax[0],
        marker="o",
        markersize=6,
        color=COL["M2"],
        label="empirical belief",
        markeredgecolor=DARK,
        markeredgewidth=0.8,
    )
    ax[0].fill_between(
        ctr,
        acc,
        ctr,
        color=COL["M2"],
        alpha=0.2,
        label="calibration gap",
    )
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("Belief confidence")
    ax[0].set_ylabel("Observed accuracy")
    ax[0].set_title("Reliability diagram")
    ax[0].locator_params(axis="x", nbins=6)
    ax[0].locator_params(axis="y", nbins=6)
    ax[0].legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    clean(ax[0])

    sns.barplot(
        x=np.round(ctr, 2),
        y=tot,
        palette=sns.color_palette("crest", len(ctr)),
        hue=np.round(ctr, 2),
        legend=False,
        alpha=0.9,
        ax=ax[1],
        edgecolor=DARK,
        linewidth=0.8,
        width=0.75,
    )
    ax[1].set_xlabel("Belief confidence")
    ax[1].set_ylabel("Frame count")
    ax[1].set_title("Calibration-bin support")
    ax[1].set_xticks(range(len(ctr)))
    ax[1].set_xticklabels([f"{x:.2f}" for x in ctr], rotation=30, ha="right")
    ax[1].locator_params(axis="y", nbins=4)
    clean(ax[1])

    save(fig, f"{pre}reliability.png")


def plot_sweep(sm, xlab, col, name, pre=""):
    use()

    xs = np.array([float(x) for x in sm])

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        layout="constrained",
    )

    for a, met, title in zip(
        ax,
        ["cov", "rdel", "seq_r"],
        [
            "Boundary coverage",
            "Failure-aware delay",
            "Sequence error",
        ],
    ):
        ys = np.array([
            sm[k][met]["mean"]
            for k in sm
        ], dtype=float)

        lo = np.array([
            sm[k][met]["lo"]
            for k in sm
        ], dtype=float)

        hi = np.array([
            sm[k][met]["hi"]
            for k in sm
        ], dtype=float)

        a.plot(xs, ys, "o-", color=col)
        a.fill_between(
            xs,
            lo,
            hi,
            color=col,
            alpha=0.18,
        )
        a.set_xlabel(xlab)
        a.set_title(title)
        clean(a)

    fig.suptitle(
        name,
        fontsize=14,
        fontweight="bold",
    )

    save(fig, f"{pre}{name.lower().replace(' ', '_')}.png")


def plot_mt(sm, pre=""):
    use()
    xs = np.array([float(x) for x in sm])

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        layout="constrained",
    )

    for a, met, title in zip(
        ax,
        ["hcov_r", "spec", "rdel"],
        [
            "Hierarchical coverage",
            "Specificity",
            "Failure-aware delay",
        ],
    ):
        ys = [sm[k][met]["mean"] for k in sm]
        lo = [sm[k][met]["lo"] for k in sm]
        hi = [sm[k][met]["hi"] for k in sm]

        a.plot(
            xs,
            ys,
            "o-",
            color=COL["MT"],
        )
        a.fill_between(
            xs,
            lo,
            hi,
            color=COL["MT"],
            alpha=0.18,
        )
        a.set_xlabel("Minimum specificity")
        a.set_title(title)
        clean(a)

    fig.suptitle(
        "Hierarchical decision sensitivity",
        fontsize=14,
        fontweight="bold",
    )

    save(fig, f"{pre}tree_sensitivity.png")


def plot_seed(raw, pre=""):
    use()

    ids = [r["seed"] for r in raw]
    names = ["M0", "M1", "M2", "M3", "MT"]

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        layout="constrained",
    )

    for k in names:
        ax[0].plot(
            ids,
            [r["cont"][k]["cov"] for r in raw],
            "o-",
            color=COL[k],
            label=k,
        )
        ax[1].plot(
            ids,
            [r["cont"][k]["rdel"] for r in raw],
            "o-",
            color=COL[k],
            label=k,
        )

    ax[0].set_title("Boundary coverage by seed")
    ax[0].set_ylabel("Coverage")
    ax[0].set_ylim(0, 1.03)
    ax[0].legend(ncol=5)
    clean(ax[0])

    ax[1].set_title("Failure-aware delay by seed")
    ax[1].set_xlabel("Evaluation seed")
    ax[1].set_ylabel("Frames")
    clean(ax[1])

    save(fig, f"{pre}seed_stability.png")


def clean_json(x):
    if isinstance(x, dict):
        return {
            str(k): clean_json(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]

    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None

    return x


def print_cont(sm):
    print()
    print(
        f"{'id':>3} | {'emit':>6} | {'per':>6} | "
        f"{'cov':>6} | {'rdel':>7} | {'hcov':>6} | "
        f"{'spec':>6} | {'seq':>6}"
    )

    for k, r in sm.items():
        def f(m):
            x = r[m]["mean"]
            return "    NA" if x is None else f"{x:6.3f}"

        print(
            f"{k:>3} | {f('emit')} | {f('per_r')} | "
            f"{f('cov')} | {f('rdel')} | "
            f"{f('hcov_r')} | {f('spec')} | "
            f"{f('seq_r')}"
        )


def plot_perseveration(sm):
    use()
    fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
    bars(ax, sm, "per_r", "Perseveration rate", (0, 1.0))
    ax.set_ylabel("Rate")
    ax.locator_params(axis="y", nbins=6)
    clean(ax)
    save(fig, "perseveration.png")


def plot_tree_specificity(sm):
    use()
    xs = np.array([float(value) for value in sm])
    fig, ax = plt.subplots(figsize=(5.0, 3.6), layout="constrained")
    ys = np.array([sm[key]["spec"]["mean"] for key in sm], dtype=float)
    lo = np.array([sm[key]["spec"]["lo"] for key in sm], dtype=float)
    hi = np.array([sm[key]["spec"]["hi"] for key in sm], dtype=float)
    ax.plot(
        xs,
        ys,
        "o-",
        color=COL["MT"],
        label="MT",
        markersize=6,
        markeredgecolor=DARK,
        markeredgewidth=0.8,
        linewidth=1.8,
    )
    ax.fill_between(xs, lo, hi, color=COL["MT"], alpha=0.2)
    ax.set_xlabel("Minimum specificity")
    ax.set_ylabel("Observed specificity")
    ax.set_title("Tree specificity sensitivity")
    ax.locator_params(axis="x", nbins=6)
    ax.locator_params(axis="y", nbins=6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
    clean(ax)
    save(fig, "tree_specificity.png")


def render_publication_artifacts(report_json):
    """regenerate publication plots from json report"""
    report = load_report_json(report_json)
    if report["profile"] != "headline":
        raise ReportValidationError(
            "non_headline output cannot populate publication evidence figures"
        )
    plot_data = report["plot_data"]
    required = ("frontier", "perseveration", "reliability", "tree_specificity")
    if any(name not in plot_data for name in required):
        raise ReportValidationError("headline JSON lacks complete publication plot data")
    plot_front(plot_data["frontier"])
    plot_perseveration(plot_data["perseveration"])
    plot_rel([{"rel": row} for row in plot_data["reliability"]])
    plot_tree_specificity(plot_data["tree_specificity"])
    return {name: ROOT / "figures" / name for name in PUBLICATION_ARTIFACTS}


def _seed_report_rows(raw):
    rows = []
    for seed_result in raw:
        for scenario in SCENARIOS:
            methods = {}
            for method in METHODS:
                metric = seed_result["scenario_cont"][scenario][method]
                methods[method] = {
                    "values": {name: metric[name] for name in METS},
                    "counts": {name: metric[name] for name in COUNT_FIELDS},
                    "compatibility": metric.get("applicability", {}),
                }
            rows.append({
                "seed": seed_result["seed"],
                "scenario": scenario,
                "ordered_test_stream_ids": seed_result["scenario_test_ids"][scenario],
                "methods": methods,
            })
    return rows


def _revision_id():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        digest = hashlib.sha256()
        for path in sorted((ROOT / "src").glob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return "source-sha256:" + digest.hexdigest()


def _report_provenance(raw, config):
    role_ids = {
        role: tuple(
            stream_id for result in raw for stream_id in result["role_stream_ids"][role]
        )
        for role in ("fit", "development", "stress")
    }
    role_ids["test"] = tuple(
        stream_id
        for result in raw
        for stream_id in (
            *result["stopping_test_ids"],
            *(item for scenario in SCENARIOS
              for item in result["scenario_test_ids"][scenario]),
        )
    )
    requirements_hash = hashlib.sha256(
        (ROOT / "requirements.txt").read_bytes()
    ).hexdigest()
    return {
        "schema_version": "1",
        "code_revision": _revision_id(),
        "environment_id": (
            f"{platform.python_implementation()}-{platform.python_version()}-"
            f"{sys.platform}-requirements-sha256:{requirements_hash}"
        ),
        "metric_definition_version": "sequential-bayes-metrics/1.0",
        "freeze_timestamp": "2025-01-01T00:00:00+00:00",
        "complete_config": config,
        "config_hash": content_hash(config),
        "predeclared_evaluation_seeds": tuple(result["seed"] for result in raw),
        "role_stream_ids": role_ids,
        "scenarios": SCENARIOS,
        "methods": METHODS,
        "numeric_rerun_tolerance": 1e-6,
    }


def _outcome_matrix():
    declarations, outcomes = [], []
    for scenario in SCENARIOS:
        for method in METHODS:
            hypothesis_id = f"{scenario}:{method}:predeclared"
            declarations.append({
                "hypothesis_id": hypothesis_id, "scenario": scenario,
                "method": method, "applicable": True,
            })
            outcomes.append({
                "hypothesis_id": hypothesis_id, "scenario": scenario,
                "method": method, "status": "inconclusive",
                "observed_direction": "mixed",
            })
    return declarations, outcomes


def run(full=False):
    """run reduced/headline profile"""
    cfg = Simcon()
    if full:
        seeds = [1103, 1201, 1301, 1409, 1511, 1601, 1709, 1801, 1901, 2003]
        n = {"sf": 250, "sd": 150, "st": 200, "cf": 60, "ct": 102}
        summary_replicates = 10_000
        profile = "headline"
    else:
        seeds = [101, 211, 307]
        n = {"sf": 80, "sd": 50, "st": 80, "cf": 25, "ct": 30}
        summary_replicates = 1_000
        profile = "reduced"

    raw = []
    for index, seed in enumerate(seeds, start=1):
        print(f"seed {index}/{len(seeds)}: {seed}")
        raw.append(one(seed, cfg, n))
    summary = summarize(raw, summary_replicates)
    seed_records = _seed_report_rows(raw)
    plot_data = clean_json({
        "frontier": summary["front"],
        "perseveration": summary["cont"],
        "reliability": [result["rel"] for result in raw],
        "tree_specificity": summary["mt"],
    })
    config = {
        "profile": profile,
        "methods": METHODS,
        "scenarios": SCENARIOS,
        "evaluation_seeds": tuple(seeds),
        "counts_per_seed": n,
        "generator": asdict(cfg),
        "bootstrap": {
            "method": "two-sided 95% paired-seed percentile bootstrap",
            "seed": 20250313,
            "replicates": 10_000,
        },
        "stopping_comparator": COMPARATOR_LABEL,
        "fixed_method_settings": {
            "lam": 0.5, "rho": 0.05, "tau": 0.8,
            "cp_thr": 0.1, "mt_min_sp": 0.25,
            "signs_per_stream": 5,
        },
    }
    provenance = _report_provenance(raw, config)
    declarations, outcomes = _outcome_matrix()
    effect_specs = ({
        "effect_id": "M2_minus_M0.boundary_coverage",
        "left_method": "M2", "right_method": "M0",
        "metric": "cov", "units": "proportion of eligible boundaries",
    },)
    RES.mkdir(exist_ok=True)
    out = RES / f"{profile}_synthetic_report.json"

    if not full:
        report = build_reduced_report(
            seed_records, provenance=provenance, paired_effects=effect_specs,
            bootstrap_seed=20250313, predeclared_outcomes=declarations,
            outcomes=outcomes, plot_data=plot_data,
        )
        write_report_json(report, out)
    else:
        artifact_paths = {
            name: ROOT / "figures" / name for name in PUBLICATION_ARTIFACTS
        }
        if any(not path.is_file() for path in artifact_paths.values()):
            # Bootstrap missing files from a JSON-only staging source.  They are
            # not evidence until hashed into the validated final report below.
            staging = RES / ".headline_plot_source.json"
            staging.write_text(json.dumps({"plot_data": plot_data}, allow_nan=False),
                               encoding="utf-8")
            staged = json.loads(staging.read_text(encoding="utf-8"))["plot_data"]
            plot_front(staged["frontier"])
            plot_perseveration(staged["perseveration"])
            plot_rel([{"rel": row} for row in staged["reliability"]])
            plot_tree_specificity(staged["tree_specificity"])
            staging.unlink()
        report = build_headline_report(
            seed_records, provenance=provenance, artifact_paths=artifact_paths,
            paired_effects=effect_specs, bootstrap_seed=20250313,
            predeclared_outcomes=declarations, outcomes=outcomes,
            plot_data=plot_data,
        )
        write_report_json(report, out)
        render_publication_artifacts(out)
        # Hash the regenerated bytes and replace the now-stale source record.
        report = build_headline_report(
            seed_records, provenance=provenance, artifact_paths=artifact_paths,
            paired_effects=effect_specs, bootstrap_seed=20250313,
            predeclared_outcomes=declarations, outcomes=outcomes,
            plot_data=plot_data,
        )
        write_report_json(report, out)
        write_report_json(report, RES / "full_synthetic_report.json")

    print_cont(summary["cont"])
    print()
    print("report:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true",
        help="run the ten-seed headline profile (default: reduced non-headline)",
    )
    args = ap.parse_args()
    run(full=args.full)


if __name__ == "__main__":
    main()
