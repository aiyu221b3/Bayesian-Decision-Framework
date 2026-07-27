"""versioned replayable reporting for paired synthetic studies.

json reports are the publication source of truth. this module deliberately
contains no experiment or plotting code, so validation and replay stay pure.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import numpy as np

from protocol import content_hash

REPORT_SCHEMA_VERSION = "sequential-bayes-report/1.0"
PROVENANCE_SCHEMA_VERSION = "1"
INTERVAL_METHOD = "two-sided 95% paired-seed percentile bootstrap"
METHODS = ("M0", "M1", "M2", "M3", "MT")
EVIDENCE_CATEGORIES = (
    "synthetic evidence", "empirical approximation", "future pilot",
)
OUTCOME_STATUSES = ("passed", "failed", "inconclusive")
OUTCOME_DIRECTIONS = ("positive", "negative", "null", "mixed")
PUBLICATION_ARTIFACTS = (
    "frontier.png", "perseveration.png", "reliability.png",
    "tree_specificity.png",
)


class ReportValidationError(ValueError):
    """invalid record report/publish error"""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, (MappingProxyType, Mapping)):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(v) for v in value)
    if isinstance(value, list):
        return [_thaw(v) for v in value]
    if isinstance(value, (frozenset, set)):
        return set(_thaw(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class ImmutableRecord(Mapping[str, Any]):
    """recursively immutable validation record"""

    fields: Mapping[str, Any]
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze(dict(self.fields)))

    def __getitem__(self, key: str) -> Any:
        if key == "schema_version":
            return self.schema_version
        return self.fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("schema_version", *self.fields))

    def __len__(self) -> int:
        return len(self.fields) + 1


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    schema_version: str
    code_revision: str
    environment_id: str
    metric_definition_version: str
    freeze_timestamp: str
    complete_config: Mapping[str, Any]
    config_hash: str
    predeclared_evaluation_seeds: tuple[int, ...]
    role_stream_ids: Mapping[str, tuple[str, ...]]
    scenarios: tuple[str, ...]
    methods: tuple[str, ...]
    numeric_rerun_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "complete_config", _freeze(self.complete_config))
        object.__setattr__(self, "role_stream_ids", _freeze(self.role_stream_ids))


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PairedEffectRecord:
    effect_id: str
    left_method: str
    right_method: str
    metric: str
    units: str
    finite_pair_count: int
    mean: float | None
    lower: float | None
    upper: float | None
    status: str
    interval_method: str
    bootstrap_seed: int
    bootstrap_replicates: int
    confidence_level: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        raise ReportValidationError(f"artifact does not exist: {target}")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _required(mapping: Mapping[str, Any], names: Sequence[str], where: str) -> None:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ReportValidationError(f"{where} missing required fields: {missing}")


def _validate_provenance(provenance: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, headline: bool) -> dict[str, Any]:
    required = (
        "schema_version", "code_revision", "environment_id",
        "metric_definition_version", "freeze_timestamp", "complete_config",
        "config_hash", "predeclared_evaluation_seeds", "role_stream_ids",
        "scenarios", "methods", "numeric_rerun_tolerance",
    )
    _required(provenance, required, "provenance")
    result = _thaw(provenance)
    if str(result["schema_version"]) != PROVENANCE_SCHEMA_VERSION:
        raise ReportValidationError("stale or unsupported provenance schema version")
    if result["config_hash"] != content_hash(result["complete_config"]):
        raise ReportValidationError("stale complete configuration hash")
    if not result["code_revision"] or not result["environment_id"]:
        raise ReportValidationError("code revision and environment provenance are required")
    tolerance = float(result["numeric_rerun_tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ReportValidationError("numeric rerun tolerance must be finite and non-negative")
    result["numeric_rerun_tolerance"] = tolerance
    seeds = tuple(int(seed) for seed in result["predeclared_evaluation_seeds"])
    observed = tuple(dict.fromkeys(int(row["seed"]) for row in records))
    if seeds != observed or len(set(seeds)) != len(seeds):
        raise ReportValidationError(
            "predeclared seeds must uniquely match seed-record first-occurrence order"
        )
    if headline and len(seeds) < 10:
        raise ReportValidationError("headline output requires at least ten unique predeclared seeds")
    methods = tuple(result["methods"])
    if methods != METHODS:
        raise ReportValidationError("provenance must retain M0, M1, M2, M3, and MT in order")
    scenarios = tuple(result["scenarios"])
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ReportValidationError("provenance scenarios must be nonempty and unique")
    expected_cells = {(seed, scenario) for seed in seeds for scenario in scenarios}
    observed_cells = {(int(row["seed"]), str(row["scenario"])) for row in records}
    if len(records) != len(observed_cells):
        raise ReportValidationError("duplicate seed/scenario records are not allowed")
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ReportValidationError(
            f"incomplete scenario matrix: missing={missing}, extra={extra}"
        )
    role_ids = result["role_stream_ids"]
    _required(role_ids, ("fit", "development", "stress", "test"), "role_stream_ids")
    flattened: list[str] = []
    for role in ("fit", "development", "stress", "test"):
        ids = tuple(str(item) for item in role_ids[role])
        if not ids:
            raise ReportValidationError(f"missing {role} stream provenance")
        flattened.extend(ids)
    if len(set(flattened)) != len(flattened):
        raise ReportValidationError("role stream IDs overlap or contain duplicates")
    expected_test = tuple(
        stream_id for row in records for stream_id in row["ordered_test_stream_ids"]
    )
    declared_test = tuple(str(item) for item in role_ids["test"])
    expected_set = set(expected_test)
    projected = tuple(item for item in declared_test if item in expected_set)
    if projected != expected_test or not expected_set.issubset(declared_test):
        raise ReportValidationError(
            "ordered test stream provenance does not contain seed records in order"
        )
    return _json_safe(result)


def _validate_seed_records(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not records:
        raise ReportValidationError("at least one seed record is required")
    retained: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, dict[str, float | int]]] = {}
    aggregate_values: dict[str, dict[str, dict[str, Any]]] = {}
    for source in records:
        _required(source, ("seed", "scenario", "ordered_test_stream_ids", "methods"), "seed record")
        ids = tuple(str(item) for item in source["ordered_test_stream_ids"])
        if not ids or len(ids) != len(set(ids)):
            raise ReportValidationError("each seed needs unique ordered test stream IDs")
        methods = source["methods"]
        if tuple(methods) != METHODS:
            raise ReportValidationError("every seed record must retain all core methods in order")
        scenario = str(source["scenario"])
        pooled.setdefault(scenario, {})
        aggregate_values.setdefault(scenario, {})
        checked_methods: dict[str, Any] = {}
        metric_keys: tuple[str, ...] | None = None
        for method in METHODS:
            row = methods[method]
            _required(row, ("values", "counts"), f"{scenario}/{method}")
            values = dict(row["values"])
            counts = dict(row["counts"])
            if not values or not counts:
                raise ReportValidationError("method values and additive counts cannot be missing")
            if metric_keys is None:
                metric_keys = tuple(values)
            elif tuple(values) != metric_keys:
                raise ReportValidationError("paired methods must expose identical metric keys")
            for name, value in counts.items():
                if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                    raise ReportValidationError(f"count {name} must be numeric")
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise ReportValidationError(f"count {name} must be finite and non-negative")
            for name, value in values.items():
                if value is not None and not isinstance(value, (int, float, np.number)):
                    raise ReportValidationError(f"metric {name} must be numeric or not estimable")
            target_counts = pooled[scenario].setdefault(method, {})
            for name, value in counts.items():
                target_counts[name] = target_counts.get(name, 0) + value
            target_values = aggregate_values[scenario].setdefault(method, {})
            for name, value in values.items():
                target_values.setdefault(name, []).append(value)
            checked_methods[method] = {
                **{
                    key: _thaw(value) for key, value in row.items()
                    if key not in {"values", "counts"}
                },
                "values": _thaw(values), "counts": _thaw(counts),
            }
        retained.append({
            **{k: _thaw(v) for k, v in source.items() if k not in {"methods", "ordered_test_stream_ids"}},
            "ordered_test_stream_ids": ids,
            "methods": checked_methods,
        })
    estimates: dict[str, Any] = {}
    for scenario, methods in aggregate_values.items():
        estimates[scenario] = {}
        for method, metrics in methods.items():
            estimates[scenario][method] = {}
            for metric, raw in metrics.items():
                finite = [float(v) for v in raw if v is not None and math.isfinite(float(v))]
                estimates[scenario][method][metric] = {
                    "mean": float(np.mean(finite)) if finite else None,
                    "finite_seed_count": len(finite),
                }
    return retained, _json_safe(pooled), estimates


def paired_seed_bootstrap(differences: Sequence[float], *, seed: int, replicates: int = 10_000, confidence_level: float = 0.95) -> dict[str, Any]:
    """compute the frozen percentile interval over finite paired seed blocks."""
    if replicates != 10_000:
        raise ReportValidationError("headline bootstrap requires exactly 10,000 resamples")
    if confidence_level != 0.95:
        raise ReportValidationError("headline confidence level must be 0.95")
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    base = {
        "finite_pair_count": int(len(values)),
        "interval_method": INTERVAL_METHOD,
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(replicates),
        "confidence_level": float(confidence_level),
    }
    if not len(values):
        return {**base, "mean": None, "lower": None, "upper": None, "status": "not_estimable"}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        **base,
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "status": "estimated",
    }


def _build_effects(records: Sequence[Mapping[str, Any]], specs: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, bootstrap_replicates: int, confidence_level: float) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for offset, spec in enumerate(specs):
        _required(spec, ("effect_id", "left_method", "right_method", "metric", "units"), "paired effect")
        left, right = spec["left_method"], spec["right_method"]
        if left not in METHODS or right not in METHODS or left == right:
            raise ReportValidationError("paired effects require two distinct core methods")
        by_seed: dict[int, list[float]] = {}
        requested_scenario = spec.get("scenario")
        for row in records:
            if requested_scenario is not None and row["scenario"] != requested_scenario:
                continue
            lv = row["methods"][left]["values"].get(spec["metric"])
            rv = row["methods"][right]["values"].get(spec["metric"])
            if lv is not None and rv is not None and math.isfinite(float(lv)) and math.isfinite(float(rv)):
                by_seed.setdefault(int(row["seed"]), []).append(float(lv) - float(rv))
        # A seed is the independent resampling block.  When an effect spans
        # scenarios, average its finite within-seed scenario differences first.
        differences = [float(np.mean(by_seed[seed])) for seed in by_seed]
        effect_id = str(spec["effect_id"])
        if effect_id in effects:
            raise ReportValidationError("paired effect IDs must be unique")
        effects[effect_id] = {
            "effect_id": effect_id,
            "left_method": left,
            "right_method": right,
            "metric": str(spec["metric"]),
            "scenario": str(spec["scenario"]) if spec.get("scenario") is not None else "all",
            "units": str(spec["units"]),
            **paired_seed_bootstrap(
                differences, seed=int(bootstrap_seed) + offset,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
            ),
        }
    return effects


def validate_outcome_retention(predeclared: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """require every applicable predeclared outcome exactly once."""
    def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
        _required(row, ("hypothesis_id", "scenario", "method"), "outcome")
        return str(row["hypothesis_id"]), str(row["scenario"]), str(row["method"])

    declarations: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    declaration_order: list[tuple[str, str, str]] = []
    for row in predeclared:
        item = key(row)
        if item in declarations:
            raise ReportValidationError("duplicate predeclared outcome")
        declarations[item] = row
        if bool(row.get("applicable", False)):
            declaration_order.append(item)
    observed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in outcomes:
        item = key(row)
        if item in observed:
            raise ReportValidationError("duplicate retained outcome")
        if item not in declarations or not bool(declarations[item].get("applicable", False)):
            raise ReportValidationError("undeclared or non-applicable outcome was retained")
        _required(row, ("status", "observed_direction"), "retained outcome")
        if row["status"] not in OUTCOME_STATUSES:
            raise ReportValidationError("outcome status must be passed, failed, or inconclusive")
        if row["observed_direction"] not in OUTCOME_DIRECTIONS:
            raise ReportValidationError("invalid observed outcome direction")
        if row["method"] not in METHODS:
            raise ReportValidationError("unknown core method in outcome")
        observed[item] = row
    if set(observed) != set(declaration_order):
        missing = set(declaration_order) - set(observed)
        extra = set(observed) - set(declaration_order)
        raise ReportValidationError(f"selective outcome omission or addition: missing={sorted(missing)}, extra={sorted(extra)}")
    return [_thaw(observed[item]) for item in declaration_order]


_PROHIBITED_ABSOLUTE = re.compile(
    r"\b(proof|proves|proven|causes?|causal|universally optimal|exact optimum|"
    r"near[- ]optimal|state[- ]of[- ]the[- ]art|real[- ]data evidence)\b",
    re.IGNORECASE,
)
_SUPERIORITY = re.compile(r"\b(superior|outperform(?:s|ed)?|better than)\b", re.IGNORECASE)


def validate_claim_artifact_registry(claims: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """validate exclusive evidence scope and adjacent audit metadata."""
    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in claims:
        _required(source, ("claim_id", "text", "evidence_categories", "artifact_id", "adjacent"), "claim")
        claim = _thaw(source)
        claim_id = str(claim["claim_id"])
        if claim_id in seen:
            raise ReportValidationError("claim IDs must be unique")
        seen.add(claim_id)
        categories = tuple(claim["evidence_categories"])
        if len(categories) != 1 or categories[0] not in EVIDENCE_CATEGORIES:
            raise ReportValidationError("each claim needs exactly one valid evidence category")
        text = str(claim["text"])
        if _PROHIBITED_ABSOLUTE.search(text):
            raise ReportValidationError("unsupported proof, causality, optimum, or real-data language")
        adjacent = claim["adjacent"]
        _required(adjacent, ("limitations", "counts", "uncertainty", "provenance"), "adjacent claim metadata")
        if not adjacent["limitations"] or not adjacent["counts"] or not adjacent["uncertainty"] or not adjacent["provenance"]:
            raise ReportValidationError("adjacent scope metadata cannot be empty")
        artifact_id = claim["artifact_id"]
        if artifact_id not in artifacts:
            raise ReportValidationError("claim references an unknown artifact")
        artifact = artifacts[artifact_id]
        _required(artifact, ("path", "sha256", "report_hash", "config_hash"), "artifact registry row")
        if not all(_is_hash(artifact[name]) for name in ("sha256", "report_hash", "config_hash")):
            raise ReportValidationError("artifact registry hashes must be SHA-256 values")
        provenance = adjacent["provenance"]
        if provenance.get("report_hash") != artifact["report_hash"] or provenance.get("config_hash") != artifact["config_hash"]:
            raise ReportValidationError("claim provenance does not match its artifact")
        if _SUPERIORITY.search(text):
            uncertainty = adjacent["uncertainty"]
            if uncertainty.get("status") != "reported" or not uncertainty.get("direction_supported", False):
                raise ReportValidationError("superiority language requires directional interval support")
        retained.append(claim)
    return retained


def _thaw(value: Any) -> Any:
    """return a json-serializable copy of a recursively frozen value."""
    if isinstance(value, ImmutableRecord):
        return {key: _thaw(value[key]) for key in value}
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_thaw(item) for item in value), key=repr)
    if isinstance(value, np.ndarray):
        return _thaw(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _artifact_records(
    artifact_paths: Mapping[str, str | Path], *, headline: bool
) -> dict[str, dict[str, str]]:
    paths = {str(name): Path(path) for name, path in artifact_paths.items()}
    if headline and set(paths) != set(PUBLICATION_ARTIFACTS):
        missing = sorted(set(PUBLICATION_ARTIFACTS) - set(paths))
        extra = sorted(set(paths) - set(PUBLICATION_ARTIFACTS))
        raise ReportValidationError(
            f"headline artifacts must be exactly the four publication plots: "
            f"missing={missing}, extra={extra}"
        )
    if not headline and paths:
        raise ReportValidationError(
            "non_headline output cannot register publication evidence artifacts"
        )
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }


def _record_hash(payload: Mapping[str, Any]) -> str:
    material = {"schema_version": REPORT_SCHEMA_VERSION, **dict(payload)}
    material.pop("report_hash", None)
    return content_hash(material)


def _build_report(
    seed_records: Sequence[Mapping[str, Any]], *,
    provenance: Mapping[str, Any], artifact_paths: Mapping[str, str | Path],
    paired_effects: Sequence[Mapping[str, Any]], bootstrap_seed: int,
    bootstrap_replicates: int, confidence_level: float, headline: bool,
    predeclared_outcomes: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    plot_data: Mapping[str, Any] | None = None,
) -> ImmutableRecord:
    retained, pooled, estimates = _validate_seed_records(seed_records)
    checked_provenance = _validate_provenance(
        provenance, retained, headline=headline
    )
    checked_outcomes = validate_outcome_retention(
        predeclared_outcomes, outcomes
    ) if predeclared_outcomes or outcomes else []
    effects = _build_effects(
        retained, paired_effects, bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    artifacts = _artifact_records(artifact_paths, headline=headline)
    payload: dict[str, Any] = {
        "profile": "headline" if headline else "non_headline",
        "evidence_category": "synthetic evidence",
        "provenance": checked_provenance,
        "per_seed": retained,
        "pooled_counts": pooled,
        "aggregate_estimates": estimates,
        "paired_effects": effects,
        "bootstrap": {
            "interval_method": INTERVAL_METHOD,
            "seed": int(bootstrap_seed),
            "replicates": int(bootstrap_replicates),
            "confidence_level": float(confidence_level),
        },
        "paired_effect_specs": [_thaw(spec) for spec in paired_effects],
        "predeclared_outcomes": [
            _thaw(row) for row in predeclared_outcomes
        ],
        "outcomes": checked_outcomes,
        "plot_data": _thaw(plot_data or {}),
        "artifacts": artifacts,
        "meta": {
            "seeds": [r["seed"] for r in retained],
            "split_stream_ids": checked_provenance.get("role_stream_ids", {}),
            "rerun_tolerance": checked_provenance.get("numeric_rerun_tolerance", 0.0),
            "artifact_paths": {name: str(row["path"]) for name, row in artifacts.items()},
            "dp_label": "empirical finite-horizon dynamic-programming approximation",
            **dict(checked_provenance),
        },
        "summary": {
            "paired_effects": effects,
            **(_thaw(plot_data or {})),
        },
    }
    payload = _json_safe(payload)
    payload["report_hash"] = _record_hash(payload)
    return ImmutableRecord(payload)


def build_headline_report(
    seed_records: Sequence[Mapping[str, Any]], *,
    provenance: Mapping[str, Any], artifact_paths: Mapping[str, str | Path],
    paired_effects: Sequence[Mapping[str, Any]], bootstrap_seed: int,
    bootstrap_replicates: int = 10_000, confidence_level: float = 0.95,
    predeclared_outcomes: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    plot_data: Mapping[str, Any] | None = None,
) -> ImmutableRecord:
    """build an immutable, publication-eligible paired multi-seed report."""
    return _build_report(
        seed_records, provenance=provenance, artifact_paths=artifact_paths,
        paired_effects=paired_effects, bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level, headline=True,
        predeclared_outcomes=predeclared_outcomes, outcomes=outcomes,
        plot_data=plot_data,
    )


def build_reduced_report(
    seed_records: Sequence[Mapping[str, Any]], *,
    provenance: Mapping[str, Any], paired_effects: Sequence[Mapping[str, Any]],
    bootstrap_seed: int, bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
    predeclared_outcomes: Sequence[Mapping[str, Any]] = (),
    outcomes: Sequence[Mapping[str, Any]] = (),
    plot_data: Mapping[str, Any] | None = None,
) -> ImmutableRecord:
    """build deterministic smoke output that is ineligible for evidence plots."""
    return _build_report(
        seed_records, provenance=provenance, artifact_paths={},
        paired_effects=paired_effects, bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level, headline=False,
        predeclared_outcomes=predeclared_outcomes, outcomes=outcomes,
        plot_data=plot_data,
    )


def validate_report(
    report: Mapping[str, Any], *,
    artifact_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """reject stale, incomplete, unsupported, or mutated report records."""
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReportValidationError("stale or unsupported report schema version")
    _required(report, (
        "profile", "evidence_category", "provenance", "per_seed",
        "pooled_counts", "aggregate_estimates", "paired_effects",
        "bootstrap", "paired_effect_specs", "predeclared_outcomes",
        "outcomes", "plot_data", "artifacts", "report_hash",
    ), "report")
    profile = report["profile"]
    if profile not in ("headline", "non_headline"):
        raise ReportValidationError("unsupported report profile")
    headline = profile == "headline"
    retained, pooled, estimates = _validate_seed_records(report["per_seed"])
    checked_provenance = _validate_provenance(
        report["provenance"], retained, headline=headline
    )
    if _json_safe(pooled) != _json_safe(report["pooled_counts"]):
        raise ReportValidationError("stale or incorrect pooled counts")
    if _json_safe(estimates) != _json_safe(report["aggregate_estimates"]):
        raise ReportValidationError("stale aggregate estimates")
    checked_outcomes = validate_outcome_retention(
        report["predeclared_outcomes"], report["outcomes"]
    ) if report["predeclared_outcomes"] or report["outcomes"] else []
    if _json_safe(checked_outcomes) != _json_safe(report["outcomes"]):
        raise ReportValidationError("stale retained outcomes")
    bootstrap = report["bootstrap"]
    _required(bootstrap, (
        "interval_method", "seed", "replicates", "confidence_level"
    ), "bootstrap")
    if bootstrap["interval_method"] != INTERVAL_METHOD:
        raise ReportValidationError("unsupported interval method")
    expected_effects = _build_effects(
        retained, report["paired_effect_specs"],
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        confidence_level=float(bootstrap["confidence_level"]),
    )
    if _json_safe(expected_effects) != _json_safe(report["paired_effects"]):
        raise ReportValidationError("stale paired effects or bootstrap bounds")
    artifacts = report["artifacts"]
    if headline and set(artifacts) != set(PUBLICATION_ARTIFACTS):
        raise ReportValidationError("headline report has incomplete publication artifacts")
    if not headline and artifacts:
        raise ReportValidationError("non_headline report cannot contain evidence artifacts")
    supplied = artifact_paths or {
        name: row["path"] for name, row in artifacts.items()
    }
    if set(supplied) != set(artifacts):
        raise ReportValidationError("artifact path set does not match report")
    for name, row in artifacts.items():
        _required(row, ("path", "sha256"), f"artifact {name}")
        path = Path(supplied[name])
        if str(path) != str(row["path"]):
            raise ReportValidationError(f"artifact path mismatch for {name}")
        if not _is_hash(row["sha256"]) or _sha256_file(path) != row["sha256"]:
            raise ReportValidationError(f"stale artifact hash for {name}")
    if checked_provenance["config_hash"] != report["provenance"]["config_hash"]:
        raise ReportValidationError("stale report configuration")
    material = _thaw(report)
    observed_hash = material.pop("report_hash")
    material.pop("schema_version")
    if not _is_hash(observed_hash) or observed_hash != _record_hash(material):
        raise ReportValidationError("stale or mutated report hash")


def replay_headline_report(
    report: Mapping[str, Any], *, seed_records: Sequence[Mapping[str, Any]],
    artifact_paths: Mapping[str, str | Path],
) -> ImmutableRecord:
    """rebuild a headline report and require replay within frozen tolerance."""
    validate_report(report, artifact_paths=artifact_paths)
    if report["profile"] != "headline":
        raise ReportValidationError("only a headline report can be headline-replayed")
    bootstrap = report["bootstrap"]
    replay = build_headline_report(
        seed_records, provenance=report["provenance"],
        artifact_paths=artifact_paths,
        paired_effects=report["paired_effect_specs"],
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        confidence_level=float(bootstrap["confidence_level"]),
        predeclared_outcomes=report["predeclared_outcomes"],
        outcomes=report["outcomes"], plot_data=report["plot_data"],
    )
    tolerance = float(report["provenance"]["numeric_rerun_tolerance"])

    def close(left: Any, right: Any, where: str = "report") -> None:
        if isinstance(left, Mapping):
            if set(left) != set(right):
                raise ReportValidationError(f"replay structure differs at {where}")
            for key in left:
                close(left[key], right[key], f"{where}.{key}")
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                raise ReportValidationError(f"replay length differs at {where}")
            for index, (a, b) in enumerate(zip(left, right)):
                close(a, b, f"{where}[{index}]")
        elif isinstance(left, (float, np.floating)):
            if not math.isclose(float(left), float(right), rel_tol=0.0,
                                abs_tol=tolerance):
                raise ReportValidationError(f"numeric replay exceeded tolerance at {where}")
        elif left != right:
            raise ReportValidationError(f"replay differs at {where}")

    close(report, replay)
    return replay


def write_report_json(report: Mapping[str, Any], path: str | Path) -> Path:
    """validate and atomically write the sole machine-readable source."""
    validate_report(report)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_thaw(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_report_json(path: str | Path, *, validate: bool = True) -> ImmutableRecord:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    schema = raw.pop("schema_version", None)
    record = ImmutableRecord(raw, schema_version=str(schema))
    if validate:
        validate_report(record)
    return record
