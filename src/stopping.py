"""empirical dp stopping-policy fitting"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from protocol import ProtocolViolationError, content_hash


COMPARATOR_LABEL = (
    "empirical finite-horizon dynamic-programming approximation"
)
STOPPING_SCHEMA_VERSION = "stopping-v2"


def myopic_tau(cw, ce):
    """myopic one-step threshold"""
    if cw < 0:
        raise ValueError("cw cannot be negative")
    if ce <= 0:
        raise ValueError("ce must be positive")
    return float(np.clip(1.0 - cw / ce, 1e-3, 1.0 - 1e-3))


def _bin(edges, x):
    b = np.searchsorted(edges, x, side="right") - 1
    return int(np.clip(b, 0, len(edges) - 2))


def _value(config: Mapping[str, Any], name: str, alias: str | None = None):
    if name in config:
        return config[name]
    if alias is not None and alias in config:
        return config[alias]
    raise ValueError(f"frozen configuration requires {name}")

def _costs(config: Mapping[str, Any]) -> tuple[float, float]:
    cw = float(_value(config, "wait_cost", "cw"))
    ce = float(_value(config, "error_cost", "ce"))
    if cw < 0:
        raise ValueError("wait_cost cannot be negative")
    if ce <= 0:
        raise ValueError("error_cost must be positive")
    return cw, ce


def _fallback(config: Mapping[str, Any]) -> dict[str, Any]:
    if "empty_bin_fallback" not in config:
        raise ValueError("empty_bin_fallback must be frozen before fitting")
    raw = config["empty_bin_fallback"]
    if not isinstance(raw, Mapping):
        raise ValueError("empty_bin_fallback must declare accuracy and transition")
    accuracy = raw.get("accuracy")
    transition = raw.get("transition")
    if accuracy not in ("bin_center", "global_fit_accuracy"):
        try:
            accuracy = float(accuracy)
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported accuracy fallback") from error
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("numeric accuracy fallback must lie in [0, 1]")
    if transition != "self":
        raise ValueError("only the frozen self transition fallback is supported")
    return {"accuracy": accuracy, "transition": transition}


def _dp_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cw, ce = _costs(config)
    nb = int(_value(config, "bin_count", "nbins"))
    if nb < 3:
        raise ValueError("bin_count must be at least 3")
    minimum_bin_visits = int(config.get("minimum_bin_visits", 1))
    minimum_transition_visits = int(config.get("minimum_transition_visits", 1))
    prior_strength = float(config.get("accuracy_prior_strength", 0.0))
    if minimum_bin_visits < 1 or minimum_transition_visits < 1:
        raise ValueError("minimum visit gates must be positive")
    if prior_strength < 0 or not np.isfinite(prior_strength):
        raise ValueError("accuracy_prior_strength must be finite and non-negative")
    return {
        "wait_cost": cw,
        "error_cost": ce,
        "bin_count": nb,
        "empty_bin_fallback": _fallback(config),
        "minimum_bin_visits": minimum_bin_visits,
        "minimum_transition_visits": minimum_transition_visits,
        "accuracy_prior_strength": prior_strength,
    }


def _stream_meta(stream: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
    manifest = stream.get("manifest")
    if manifest is not None:
        stream_id = str(manifest.stream_id)
        role = str(manifest.role)
        seed = int(manifest.seed)
        if "stream_id" in stream and str(stream["stream_id"]) != stream_id:
            raise ProtocolViolationError("stream and manifest IDs disagree")
        if "role" in stream and str(stream["role"]) != role:
            raise ProtocolViolationError("stream and manifest roles disagree")
        return stream_id, role, seed
    stream_id = str(stream["stream_id"]) if "stream_id" in stream else None
    role = str(stream["role"]) if "role" in stream else None
    seed = int(stream.get("seed", 0))
    return stream_id, role, seed


def _require_role(streams, role: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    rows = tuple(streams)
    if not rows:
        raise ValueError(f"need at least one {role} stream")
    ids, seeds = [], []
    for stream in rows:
        stream_id, observed_role, seed = _stream_meta(stream)
        if stream_id is None or observed_role is None:
            raise ProtocolViolationError(
                f"{role} use requires stream IDs and protocol role records"
            )
        if observed_role != role:
            raise ProtocolViolationError(
                f"{role} operation received {observed_role} stream {stream_id}"
            )
        ids.append(stream_id)
        seeds.append(seed)
    if len(set(ids)) != len(ids):
        raise ProtocolViolationError(f"duplicate {role} stream ID")
    return tuple(ids), tuple(seeds)


def _arrays(stream: Mapping[str, Any]):
    bm = np.asarray(stream["bmax"], dtype=float)
    pred = np.asarray(stream["pred"], dtype=int)
    if bm.ndim != 1 or len(bm) == 0 or bm.shape != pred.shape:
        raise ValueError("bmax and pred must be matching nonempty 1D arrays")
    if not np.all(np.isfinite(bm)) or np.any((bm < 0.0) | (bm > 1.0)):
        raise ValueError("bmax values must be finite and lie in [0, 1]")
    return bm, pred, int(stream["truth"])

def commit_time(bm, pol):
    bm = np.asarray(bm, dtype=float)
    if bm.ndim != 1 or len(bm) == 0:
        raise ValueError("bm must be a nonempty 1D array")
    for t, x in enumerate(bm):
        if pol(t, float(x)):
            return t
    return len(bm) - 1


def threshold_decide(tau):
    tau = float(tau)
    if not 0 <= tau <= 1:
        raise ValueError("tau must lie in [0, 1]")
    return lambda t, x: x >= tau


def _score(streams, policy, cw: float, ce: float) -> dict[str, Any]:
    rows = tuple(streams)
    if not rows:
        raise ValueError("need at least one stream")
    correct = latency_sum = 0
    for stream in rows:
        bm, pred, truth = _arrays(stream)
        t = commit_time(bm, policy)
        correct += int(pred[t] == truth)
        latency_sum += int(t)
    n = len(rows)
    errors = n - correct
    stop_cost_sum = ce * errors
    wait_cost_sum = cw * latency_sum
    risk_sum = stop_cost_sum + wait_cost_sum
    return {
        "risk": float(risk_sum / n),
        "accuracy": float(correct / n),
        "acc": float(correct / n),
        "latency": float(latency_sum / n),
        "stream_count": n,
        "n": n,
        "correct_count": int(correct),
        "error_count": int(errors),
        "latency_sum": int(latency_sum),
        "stop_cost_sum": float(stop_cost_sum),
        "wait_cost_sum": float(wait_cost_sum),
        "risk_sum": float(risk_sum),
    }


def evaluate(ss, pol, cw, ce):
    """scorer with overlap rejection"""
    rows = tuple(ss)
    test_ids = tuple(
        stream_id for stream_id, _, _ in map(_stream_meta, rows)
        if stream_id is not None
    )
    fit_ids = tuple(getattr(pol, "fit_stream_ids", ()))
    overlap = set(test_ids) & set(fit_ids)
    if overlap:
        raise ProtocolViolationError(
            f"fit/test identifier overlap: {sorted(overlap)}"
        )
    scored = _score(rows, pol, float(cw), float(ce))
    return {key: scored[key] for key in ("risk", "acc", "latency", "n")}


def _accuracy_fallback(spec, centers, hit, total):
    if spec == "bin_center":
        return centers.copy()
    if spec == "global_fit_accuracy":
        denominator = float(total.sum())
        value = float(hit.sum() / denominator) if denominator else 0.5
        return np.full_like(centers, value)
    return np.full_like(centers, float(spec))


def fit_empirical_dp(fit_streams, frozen_config):
    """fit empirical dp on fit only"""
    streams = tuple(fit_streams)
    fit_ids, fit_seeds = _require_role(streams, "fit")
    config = _dp_config(frozen_config)
    cw, ce = config["wait_cost"], config["error_cost"]
    nb = config["bin_count"]
    arrays = [_arrays(stream) for stream in streams]
    horizon = max(len(bm) for bm, _, _ in arrays)
    edges = np.linspace(0.0, 1.0, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    transitions = np.zeros((horizon, nb, nb), dtype=np.int64)
    occupancy = np.zeros((horizon, nb), dtype=np.int64)
    hit = np.zeros(nb, dtype=np.int64)
    total = np.zeros(nb, dtype=np.int64)
    for bm, pred, truth in arrays:
        for t, x in enumerate(bm):
            b = _bin(edges, x)
            occupancy[t, b] += 1
            total[b] += 1
            hit[b] += int(pred[t] == truth)
            if t + 1 < len(bm):
                transitions[t, b, _bin(edges, bm[t + 1])] += 1

    fallback_accuracy = _accuracy_fallback(
        config["empty_bin_fallback"]["accuracy"], centers, hit, total
    )
    minimum = config["minimum_bin_visits"]
    prior_strength = config["accuracy_prior_strength"]
    usable = total >= minimum
    accuracy = fallback_accuracy.copy()
    accuracy[usable] = (
        hit[usable] + prior_strength * fallback_accuracy[usable]
    ) / (total[usable] + prior_strength)
    raw_accuracy = [
        float(hit[index] / total[index]) if total[index] else None
        for index in range(nb)
    ]

    stop_cost = ce * (1.0 - accuracy)
    value_next = stop_cost.copy()
    region = np.ones((horizon, nb), dtype=bool)
    wait_cost = [[None for _ in range(nb)] for _ in range(horizon)]
    bellman_margin = [[None for _ in range(nb)] for _ in range(horizon)]
    transition_visits = transitions.sum(axis=2)
    transition_fallbacks = []
    for t in range(horizon - 2, -1, -1):
        value = np.empty(nb, dtype=float)
        for b in range(nb):
            visits = int(transition_visits[t, b])
            if visits >= config["minimum_transition_visits"]:
                future = (transitions[t, b] / visits) @ value_next
            else:
                future = value_next[b]
                transition_fallbacks.append({"time": t, "bin": b, "visits": visits})
            go = float(cw + future)
            margin = float(go - stop_cost[b])
            region[t, b] = bool(stop_cost[b] <= go)
            value[b] = min(float(stop_cost[b]), go)
            wait_cost[t][b] = go
            bellman_margin[t][b] = margin
        value_next = value

    diagnostics = {
        "fit_stream_count": len(streams),
        "fit_seed_count": len(set(fit_seeds)),
        "horizon": horizon,
        "bin_count": nb,
        "bin_visit_counts": total.tolist(),
        "bin_correct_counts": hit.tolist(),
        "raw_bin_accuracy": raw_accuracy,
        "effective_bin_accuracy": accuracy.tolist(),
        "occupancy_counts": occupancy.tolist(),
        "transition_visit_counts": transition_visits.tolist(),
        "transition_counts": transitions.tolist(),
        "fallback_usage": {
            "accuracy_bin_indices": np.flatnonzero(~usable).astype(int).tolist(),
            "accuracy_bin_count": int((~usable).sum()),
            "transition_states": transition_fallbacks,
            "transition_state_count": len(transition_fallbacks),
        },
        "stop_cost": stop_cost.tolist(),
        "wait_cost": wait_cost,
        "bellman_margin_wait_minus_stop": bellman_margin,
    }
    record = {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "kind": "empirical_dp",
        "method_id": "empirical_dp",
        "label": COMPARATOR_LABEL,
        "config": config,
        "fit_stream_ids": list(fit_ids),
        "fit_seed_ids": sorted(set(fit_seeds)),
        "edges": edges.tolist(),
        "stop_region": region.tolist(),
        "diagnostics": diagnostics,
    }
    record["policy_hash"] = content_hash(record)
    return record


def fit_empirical_dp_sensitivity(fit_streams, frozen_configs, fit_sizes=None):
    """fit bin-count/fit-size variants"""
    streams = tuple(fit_streams)
    sizes = tuple(fit_sizes) if fit_sizes is not None else (len(streams),)
    if not sizes or any(int(size) < 1 or int(size) > len(streams) for size in sizes):
        raise ValueError("fit_sizes must be positive prefixes of fit_streams")
    records = []
    for config in tuple(frozen_configs):
        for size in sizes:
            fitted = fit_empirical_dp(streams[:int(size)], config)
            records.append({
                "fit_size": int(size),
                "bin_count": fitted["config"]["bin_count"],
                "policy": fitted,
            })
    return records


def _selection_spec(candidates):
    if isinstance(candidates, Mapping):
        thresholds = candidates.get("thresholds", candidates.get("candidates"))
        if thresholds is None:
            raise ValueError("candidates must contain thresholds")
        cw, ce = _costs(candidates)
        return tuple(float(x) for x in thresholds), cw, ce
    rows = tuple(candidates)
    if not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("candidate records must freeze threshold and costs")
    thresholds = tuple(float(row["threshold"]) for row in rows)
    costs = {_costs(row) for row in rows}
    if len(costs) != 1:
        raise ValueError("all threshold candidates must use identical frozen costs")
    cw, ce = costs.pop()
    return thresholds, cw, ce


def select_fixed_threshold(development_streams, candidates):
    """select threshold from dev streams"""
    streams = tuple(development_streams)
    development_ids, development_seeds = _require_role(streams, "development")
    thresholds, cw, ce = _selection_spec(candidates)
    if not thresholds or any(not 0.0 <= tau <= 1.0 for tau in thresholds):
        raise ValueError("threshold candidates must be nonempty and lie in [0, 1]")
    scores = []
    for tau in thresholds:
        metrics = _score(streams, threshold_decide(tau), cw, ce)
        scores.append({"threshold": tau, "metrics": metrics})
    selected = min(
        scores,
        key=lambda row: (
            row["metrics"]["risk"], row["metrics"]["latency"],
            -row["metrics"]["accuracy"], row["threshold"],
        ),
    )
    record = {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "kind": "fixed_threshold",
        "method_id": "fixed",
        "label": "development-selected fixed threshold",
        "threshold": selected["threshold"],
        "config": {"wait_cost": cw, "error_cost": ce},
        "development_stream_ids": list(development_ids),
        "development_seed_ids": sorted(set(development_seeds)),
        "candidate_scores": scores,
    }
    record["policy_hash"] = content_hash(record)
    return record

def _baseline_record(kind, method_id, label, config, threshold=None):
    cw, ce = _costs(config)
    record = {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "kind": kind,
        "method_id": method_id,
        "label": label,
        "config": {"wait_cost": cw, "error_cost": ce},
    }
    if threshold is not None:
        record["threshold"] = float(threshold)
    record["policy_hash"] = content_hash(record)
    return record


def freeze_stopping_policies(frozen_config, empirical_dp, fixed_threshold):
    """assemble stopping comparators"""
    cw, ce = _costs(frozen_config)
    expected = (cw, ce)
    for name, record in (("empirical DP", empirical_dp), ("fixed", fixed_threshold)):
        if _costs(record["config"]) != expected:
            raise ProtocolViolationError(f"{name} policy costs differ from frozen costs")
    policies = {
        "immediate": _baseline_record(
            "immediate_commit", "immediate", "immediate-commit baseline", frozen_config
        ),
        "myopic": _baseline_record(
            "threshold", "myopic", "myopic threshold", frozen_config,
            myopic_tau(cw, ce),
        ),
        "fixed": deepcopy(fixed_threshold),
        "empirical_dp": deepcopy(empirical_dp),
    }
    bundle = {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "config": {"wait_cost": cw, "error_cost": ce},
        "policies": policies,
        "fit_stream_ids": list(empirical_dp["fit_stream_ids"]),
        "development_stream_ids": list(fixed_threshold["development_stream_ids"]),
    }
    if set(bundle["fit_stream_ids"]) & set(bundle["development_stream_ids"]):
        raise ProtocolViolationError("fit/development identifier overlap")
    bundle["bundle_hash"] = content_hash(bundle)
    return bundle


def _record_policy(record):
    kind = record["kind"]
    if kind == "immediate_commit":
        return lambda t, x: t == 0
    if kind == "threshold" or kind == "fixed_threshold":
        return threshold_decide(record["threshold"])
    if kind == "empirical_dp":
        region = np.asarray(record["stop_region"], dtype=bool)
        edges = np.asarray(record["edges"], dtype=float)
        return lambda t, x: bool(region[min(int(t), len(region) - 1), _bin(edges, x)])
    raise ValueError(f"unknown frozen stopping policy kind: {kind}")


def _difference(left, right):
    return {
        "risk": float(left["risk"] - right["risk"]),
        "accuracy": float(left["accuracy"] - right["accuracy"]),
        "latency": float(left["latency"] - right["latency"]),
        "units": {
            "risk": "cost per stream",
            "accuracy": "proportion correct",
            "latency": "frames per stream",
        },
    }


def evaluate_stopping(test_streams, frozen_policies):
    """eval frozen policies"""
    streams = tuple(test_streams)
    test_ids, seeds = _require_role(streams, "test")
    bundle = deepcopy(frozen_policies)
    if bundle.get("schema_version") != STOPPING_SCHEMA_VERSION:
        raise ProtocolViolationError("unsupported or missing frozen stopping schema")
    supplied_hash = bundle.pop("bundle_hash", None)
    if supplied_hash != content_hash(bundle):
        raise ProtocolViolationError("stopping policies changed after freeze")
    policies = bundle.get("policies", {})
    required = ("immediate", "myopic", "fixed", "empirical_dp")
    if tuple(policies) != required:
        raise ProtocolViolationError("all frozen stopping methods are required in order")
    training_ids = set(bundle.get("fit_stream_ids", ())) | set(
        bundle.get("development_stream_ids", ())
    )
    overlap = training_ids & set(test_ids)
    if overlap:
        raise ProtocolViolationError(
            f"fit/development/test identifier overlap: {sorted(overlap)}"
        )
    cw, ce = _costs(bundle["config"])

    groups: list[tuple[int, list[Mapping[str, Any]]]] = []
    group_index: dict[int, int] = {}
    for stream, seed in zip(streams, seeds):
        if seed not in group_index:
            group_index[seed] = len(groups)
            groups.append((seed, []))
        groups[group_index[seed]][1].append(stream)

    per_seed = []
    for seed, seed_streams in groups:
        ordered_ids = [_stream_meta(stream)[0] for stream in seed_streams]
        method_rows = {}
        for method_id in required:
            record = policies[method_id]
            metrics = _score(seed_streams, _record_policy(record), cw, ce)
            metrics["label"] = record["label"]
            metrics["ordered_test_stream_ids"] = list(ordered_ids)
            method_rows[method_id] = metrics
        dp = method_rows["empirical_dp"]
        paired = {
            "dp_minus_myopic": _difference(dp, method_rows["myopic"]),
            "dp_minus_fixed": _difference(dp, method_rows["fixed"]),
        }
        per_seed.append({
            "seed": int(seed),
            "ordered_test_stream_ids": list(ordered_ids),
            "methods": method_rows,
            "paired_differences": paired,
        })

    pooled = {}
    for method_id in required:
        rows = [row["methods"][method_id] for row in per_seed]
        counts = {
            key: sum(row[key] for row in rows)
            for key in (
                "stream_count", "correct_count", "error_count", "latency_sum",
                "stop_cost_sum", "wait_cost_sum", "risk_sum",
            )
        }
        n = counts["stream_count"]
        pooled[method_id] = {
            "label": policies[method_id]["label"],
            "risk": float(counts["risk_sum"] / n),
            "accuracy": float(counts["correct_count"] / n),
            "acc": float(counts["correct_count"] / n),
            "latency": float(counts["latency_sum"] / n),
            **counts,
            "n": n,
            "ordered_test_stream_ids": list(test_ids),
        }
    paired_summary = {}
    for key in ("dp_minus_myopic", "dp_minus_fixed"):
        paired_summary[key] = {
            metric: float(np.mean([
                row["paired_differences"][key][metric] for row in per_seed
            ]))
            for metric in ("risk", "accuracy", "latency")
        }
        paired_summary[key]["seed_count"] = len(per_seed)
    return {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "ordered_test_stream_ids": list(test_ids),
        "per_seed": per_seed,
        "methods": pooled,
        "paired_differences": paired_summary,
    }

@dataclass
class DPPol:
    """compat callable for empirical dp"""

    reg: np.ndarray
    edges: np.ndarray
    fit_stream_ids: tuple[str, ...] = field(default_factory=tuple)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, t, x):
        ti = min(int(t), self.reg.shape[0] - 1)
        return bool(self.reg[ti, _bin(self.edges, float(x))])


def _legacy_records(streams, role):
    records = []
    for index, stream in enumerate(streams):
        row = dict(stream)
        row.setdefault("stream_id", f"legacy-{role}-{index}")
        row.setdefault("role", role)
        row.setdefault("seed", 0)
        records.append(row)
    return records


def fit_dp(fit, cw, ce, nb=25):
    """compat wrapper for fit_empirical_dp"""
    rows = tuple(fit)
    strict = any(_stream_meta(stream)[1] is not None for stream in rows)
    records = rows if strict else _legacy_records(rows, "fit")
    fitted = fit_empirical_dp(records, {
        "wait_cost": cw,
        "error_cost": ce,
        "bin_count": nb,
        "empty_bin_fallback": {"accuracy": "bin_center", "transition": "self"},
        "minimum_bin_visits": 1,
        "minimum_transition_visits": 1,
        "accuracy_prior_strength": 0.0,
    })
    return DPPol(
        reg=np.asarray(fitted["stop_region"], dtype=bool),
        edges=np.asarray(fitted["edges"], dtype=float),
        fit_stream_ids=tuple(
            stream_id for stream_id, _, _ in map(_stream_meta, rows)
            if stream_id is not None
        ),
        diagnostics=fitted["diagnostics"],
    )


def best_fixed_threshold(dev, cw, ce, grid=None):
    """compat wrapper for historical return shape"""
    if grid is None:
        grid = np.linspace(0.30, 0.99, 50)
    rows = tuple(dev)
    strict = any(_stream_meta(stream)[1] is not None for stream in rows)
    records = rows if strict else _legacy_records(rows, "development")
    selected = select_fixed_threshold(records, {
        "thresholds": list(grid), "wait_cost": cw, "error_cost": ce,
    })
    metrics = selected["candidate_scores"][[
        row["threshold"] for row in selected["candidate_scores"]
    ].index(selected["threshold"])]["metrics"]
    return selected["threshold"], {
        key: metrics[key] for key in ("risk", "acc", "latency", "n")
    }


def calibration(ss, nb=15):
    if nb < 2:
        raise ValueError("nb must be at least 2")
    edges = np.linspace(0.0, 1.0, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hit = np.zeros(nb, dtype=float)
    total = np.zeros(nb, dtype=float)
    for stream in ss:
        bm, pred, truth = _arrays(stream)
        for x, guess in zip(bm, pred):
            b = _bin(edges, x)
            total[b] += 1
            hit[b] += int(guess == truth)
    accuracy = np.divide(
        hit, total, out=np.full(nb, np.nan), where=total > 0
    )
    return centers, accuracy, total


def dp_optimal(fit, cw, ce, nbins=25):
    """historical api name"""
    policy = fit_dp(fit, cw, ce, nb=nbins)
    return policy, {"label": COMPARATOR_LABEL}
