"""versioned metrics for flat/hierarchical emissions"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np


METRIC_SCHEMA_VERSION = "sequential-bayes-metrics/1.0"
APPLICABLE = "applicable"
NOT_APPLICABLE_NONLEAF = "not_applicable_nonleaf"
NOT_ESTIMABLE = "not_estimable_no_opportunity"
ESTIMATED = "estimated"
HIT = "hit"
RIGHT_CENSORED = "right_censored"

COUNT_FIELDS = (
    "n_emit", "rep_n", "rep_d", "err_n", "err_d", "per_n", "per_d",
    "seq_n", "seq_d", "bnd_n", "hit_n", "miss_n", "rdel_sum",
    "hdel_sum", "hdel_n", "hcov_n", "hcov_d", "card_sum", "card_n",
    "spec_sum", "spec_n",
)
RATE_FIELDS = (
    "rep_r", "err_r", "per_r", "seq_r", "cov", "rdel", "hdel",
    "hcov_r", "card", "spec",
)
METRIC_FIELDS = COUNT_FIELDS + RATE_FIELDS + ("flat_ok",)


def safe_rate(numerator: float, denominator: float) -> float:
    """rate when denom > 0"""
    return float(numerator / denominator) if denominator > 0 else float("nan")


rate = safe_rate

def node(identity: Any) -> frozenset[int]:
    """immutable leaf support from emission"""
    if isinstance(identity, (set, frozenset)):
        if any(isinstance(value, bool) for value in identity):
            raise TypeError("leaf identifiers must be integers, not booleans")
        support = frozenset(int(value) for value in identity)
    elif isinstance(identity, (int, np.integer)) and not isinstance(identity, bool):
        support = frozenset((int(identity),))
    else:
        raise TypeError("identity must be an integer leaf or a set of leaves")
    if not support:
        raise ValueError("an emitted node cannot be empty")
    return support


def leaf(identity: Any) -> int | None:
    """unwrap singleton support"""
    support = node(identity)
    return next(iter(support)) if len(support) == 1 else None


def collapse(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """levenshtein distance"""
    n, m = len(a), len(b)
    distances = np.zeros((n + 1, m + 1), dtype=int)
    distances[:, 0] = np.arange(n + 1)
    distances[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = int(a[i - 1] != b[j - 1])
            distances[i, j] = min(
                distances[i - 1, j] + 1,
                distances[i, j - 1] + 1,
                distances[i - 1, j - 1] + substitution,
            )
    return int(distances[n, m])


class _ImmutableRow(Mapping[str, Any]):
    _KEYS: ClassVar[tuple[str, ...]]

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    def to_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}


@dataclass(frozen=True, slots=True)
class RawEmissionRow(_ImmutableRow):
    """raw emission row"""

    stream_index: int
    emission_index: int
    time: int
    identity: int | frozenset[int]
    node: frozenset[int]
    truth: int
    segment_index: int
    is_leaf: bool
    flat_identity: int | None
    covered: bool

    _KEYS = (
        "stream_index", "emission_index", "time", "identity", "node",
        "truth", "segment_index", "is_leaf", "flat_identity", "covered",
    )


@dataclass(frozen=True, slots=True)
class HierarchyEmissionRow(_ImmutableRow):
    """hierarchy scoring row"""

    stream_index: int
    emission_index: int
    time: int
    node: frozenset[int]
    truth: int
    segment_index: int
    covered: bool
    cardinality: int
    specificity: float

    _KEYS = (
        "stream_index", "emission_index", "time", "node", "truth",
        "segment_index", "covered", "cardinality", "specificity",
    )


@dataclass(frozen=True, slots=True)
class BoundaryRow(_ImmutableRow):
    """boundary row: hit or right-censored miss"""

    stream_index: int
    segment_index: int
    onset: int
    end: int
    truth: int
    status: str
    hit: bool
    emission_time: int | None
    delay: int
    censored: bool

    _KEYS = (
        "stream_index", "segment_index", "onset", "end", "truth",
        "status", "hit", "emission_time", "delay", "censored",
    )


@dataclass(frozen=True, slots=True)
class MetricRecord(Mapping[str, Any]):
    """immutable metric record with counts & rates"""

    n_emit: int
    rep_n: int
    rep_d: int
    err_n: int
    err_d: int
    per_n: int
    per_d: int
    seq_n: int
    seq_d: int
    bnd_n: int
    hit_n: int
    miss_n: int
    rdel_sum: float
    hdel_sum: float
    hdel_n: int
    hcov_n: int
    hcov_d: int
    card_sum: float
    card_n: int
    spec_sum: float
    spec_n: int
    rep_r: float
    err_r: float
    per_r: float
    seq_r: float
    cov: float
    rdel: float
    hdel: float
    hcov_r: float
    card: float
    spec: float
    flat_ok: bool
    applicability: Mapping[str, str]
    status: Mapping[str, str]
    _raw_emission_rows: tuple[RawEmissionRow, ...]
    _hierarchy_rows: tuple[HierarchyEmissionRow, ...]
    _boundary_rows: tuple[BoundaryRow, ...]
    schema_version: str = METRIC_SCHEMA_VERSION

    _KEYS: ClassVar[tuple[str, ...]] = METRIC_FIELDS
    _DETAIL_KEYS: ClassVar[tuple[str, ...]] = (
        "schema_version", "applicability", "status", "raw_emission_rows",
        "raw_emissions", "hierarchy_rows", "boundary_rows",
    )

    def __getitem__(self, key: str) -> Any:
        if key in self._KEYS:
            return getattr(self, key)
        if key == "schema_version":
            return self.schema_version
        if key in ("raw_emission_rows", "raw_emissions"):
            return [row.to_dict() for row in self._raw_emission_rows]
        if key == "hierarchy_rows":
            return [row.to_dict() for row in self._hierarchy_rows]
        if key == "boundary_rows":
            return [row.to_dict() for row in self._boundary_rows]
        if key in ("applicability", "status"):
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        # The stable estimand projection remains exactly the Property 5 schema.
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)

    @property
    def raw_emission_rows(self) -> tuple[RawEmissionRow, ...]:
        return self._raw_emission_rows

    @property
    def hierarchy_rows(self) -> tuple[HierarchyEmissionRow, ...]:
        return self._hierarchy_rows

    @property
    def boundary_rows(self) -> tuple[BoundaryRow, ...]:
        return self._boundary_rows

    def to_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        result = {key: self[key] for key in self}
        if include_details:
            def serializable(row: _ImmutableRow) -> dict[str, Any]:
                values = row.to_dict()
                for key, value in values.items():
                    if isinstance(value, frozenset):
                        values[key] = sorted(value)
                return values

            result.update({
                "schema_version": self.schema_version,
                "applicability": dict(self.applicability),
                "status": dict(self.status),
                "raw_emission_rows": [serializable(row) for row in self._raw_emission_rows],
                "hierarchy_rows": [serializable(row) for row in self._hierarchy_rows],
                "boundary_rows": [serializable(row) for row in self._boundary_rows],
            })
        return result


def _metric_state(flat_ok: bool, denominator: float, *, flat_only: bool = False) -> tuple[str, str]:
    if flat_only and not flat_ok:
        return NOT_APPLICABLE_NONLEAF, NOT_APPLICABLE_NONLEAF
    return APPLICABLE, ESTIMATED if denominator > 0 else NOT_ESTIMABLE


def _make_record(
    counts: Mapping[str, float],
    flat_ok: bool,
    raw_rows: tuple[RawEmissionRow, ...] = (),
    hierarchy_rows: tuple[HierarchyEmissionRow, ...] = (),
    boundary_rows: tuple[BoundaryRow, ...] = (),
) -> MetricRecord:
    values = {field: counts.get(field, 0) for field in COUNT_FIELDS}
    flat_rate = lambda numerator, denominator: (
        safe_rate(numerator, denominator) if flat_ok else float("nan")
    )
    rates = {
        "rep_r": safe_rate(values["rep_n"], values["rep_d"]),
        "err_r": flat_rate(values["err_n"], values["err_d"]),
        "per_r": flat_rate(values["per_n"], values["per_d"]),
        "seq_r": flat_rate(values["seq_n"], values["seq_d"]),
        "cov": safe_rate(values["hit_n"], values["bnd_n"]),
        "rdel": safe_rate(values["rdel_sum"], values["bnd_n"]),
        "hdel": safe_rate(values["hdel_sum"], values["hdel_n"]),
        "hcov_r": safe_rate(values["hcov_n"], values["hcov_d"]),
        "card": safe_rate(values["card_sum"], values["card_n"]),
        "spec": safe_rate(values["spec_sum"], values["spec_n"]),
    }
    definitions = {
        "repeat": (values["rep_d"], False),
        "emission_error": (values["err_d"], True),
        "perseveration": (values["per_d"], True),
        "sequence_edit": (values["seq_d"], True),
        "boundary_coverage": (values["bnd_n"], False),
        "restricted_delay": (values["bnd_n"], False),
        "conditional_hit_delay": (values["hdel_n"], False),
        "hierarchical_coverage": (values["hcov_d"], False),
        "cardinality": (values["card_n"], False),
        "specificity": (values["spec_n"], False),
    }
    states = {
        name: _metric_state(flat_ok, denominator, flat_only=flat_only)
        for name, (denominator, flat_only) in definitions.items()
    }
    applicability = MappingProxyType({name: state[0] for name, state in states.items()})
    status = MappingProxyType({name: state[1] for name, state in states.items()})
    integer_fields = set(COUNT_FIELDS) - {"rdel_sum", "hdel_sum", "card_sum", "spec_sum"}
    normalized = {
        key: int(value) if key in integer_fields else float(value)
        for key, value in values.items()
    }
    return MetricRecord(
        **normalized, **rates, flat_ok=bool(flat_ok),
        applicability=applicability, status=status,
        _raw_emission_rows=raw_rows,
        _hierarchy_rows=hierarchy_rows,
        _boundary_rows=boundary_rows,
    )


def _validate_stream(
    z: Sequence[int], labels: Sequence[int], onsets: Sequence[int], K: int
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    truth = np.asarray(z)
    if truth.ndim != 1 or len(truth) == 0:
        raise ValueError("z must be a non-empty one-dimensional truth sequence")
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 2:
        raise ValueError("K must be an integer at least 2")
    label_values = tuple(int(value) for value in labels)
    onset_values = tuple(int(value) for value in onsets)
    if len(label_values) != len(onset_values) or not onset_values:
        raise ValueError("labels and onsets must be non-empty and have equal length")
    if onset_values[0] != 0 or any(
        left >= right for left, right in zip(onset_values, onset_values[1:])
    ):
        raise ValueError("onsets must start at zero and be strictly increasing")
    if onset_values[-1] >= len(truth):
        raise ValueError("every onset must identify a frame in z")
    if np.any(np.asarray(truth, dtype=int) < 0) or np.any(np.asarray(truth, dtype=int) >= K):
        raise ValueError("truth labels must lie in [0, K)")
    if any(value < 0 or value >= K for value in label_values):
        raise ValueError("segment labels must lie in [0, K)")
    return np.asarray(truth, dtype=int), label_values, onset_values


def _identity(identity: Any, K: int) -> tuple[int | frozenset[int], frozenset[int]]:
    support = node(identity)
    if any(value < 0 or value >= K for value in support):
        raise ValueError("emitted leaf identifiers must lie in [0, K)")
    exact = int(identity) if isinstance(identity, (int, np.integer)) else support
    return exact, support


def emission_metrics(
    emissions: Sequence[tuple[int, Any]],
    z: Sequence[int],
    labels: Sequence[int],
    onsets: Sequence[int],
    K: int | None = None,
    *,
    stream_index: int = 0,
) -> MetricRecord:
    """score stream without dropping misses"""
    if K is None:
        candidates = [int(value) for value in np.asarray(z).reshape(-1)]
        candidates.extend(int(value) for value in labels)
        for _, identity in emissions:
            candidates.extend(node(identity))
        K = max(candidates, default=1) + 1
        K = max(K, 2)
    truth, label_values, onset_values = _validate_stream(z, labels, onsets, K)
    if isinstance(stream_index, bool) or int(stream_index) < 0:
        raise ValueError("stream_index must be a non-negative integer")

    raw_rows: list[RawEmissionRow] = []
    hierarchy_rows: list[HierarchyEmissionRow] = []
    for emission_index, item in enumerate(emissions):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("each emission must be a (time, identity) pair")
        time, identity = item
        if isinstance(time, bool) or not isinstance(time, (int, np.integer)):
            raise TypeError("emission time must be an integer frame index")
        time = int(time)
        if time < 0 or time >= len(truth):
            raise ValueError("emission time must lie within the stream")
        exact, support = _identity(identity, int(K))
        segment_index = int(np.searchsorted(onset_values, time, side="right") - 1)
        active_truth = int(truth[time])
        flat_identity = next(iter(support)) if len(support) == 1 else None
        covered = active_truth in support
        raw_rows.append(RawEmissionRow(
            int(stream_index), emission_index, time, exact, support, active_truth,
            segment_index, flat_identity is not None, flat_identity, covered,
        ))
        hierarchy_rows.append(HierarchyEmissionRow(
            int(stream_index), emission_index, time, support, active_truth,
            segment_index, covered, len(support),
            float((int(K) - len(support)) / (int(K) - 1)),
        ))

    flat_ok = all(row.is_leaf for row in raw_rows)
    supports = [row.node for row in raw_rows]
    counts: dict[str, float] = {field: 0 for field in COUNT_FIELDS}
    counts["n_emit"] = len(raw_rows)
    counts["rep_d"] = max(len(raw_rows) - 1, 0)
    counts["rep_n"] = sum(left == right for left, right in zip(supports, supports[1:]))
    counts["hcov_n"] = sum(row.covered for row in hierarchy_rows)
    counts["hcov_d"] = len(hierarchy_rows)
    counts["card_sum"] = sum(row.cardinality for row in hierarchy_rows)
    counts["card_n"] = len(hierarchy_rows)
    counts["spec_sum"] = sum(row.specificity for row in hierarchy_rows)
    counts["spec_n"] = len(hierarchy_rows)

    if flat_ok:
        counts["err_d"] = len(raw_rows)
        counts["err_n"] = sum(row.flat_identity != row.truth for row in raw_rows)
        eligible_perseveration = [row for row in raw_rows if row.segment_index > 0]
        counts["per_d"] = len(eligible_perseveration)
        counts["per_n"] = sum(
            row.flat_identity == label_values[row.segment_index - 1]
            and row.flat_identity != row.truth
            for row in eligible_perseveration
        )
        predicted = collapse([row.flat_identity for row in raw_rows])
        counts["seq_n"] = levenshtein(predicted, label_values)
        counts["seq_d"] = len(label_values)

    boundary_rows: list[BoundaryRow] = []
    for segment_index in range(1, len(onset_values)):
        onset = onset_values[segment_index]
        end = onset_values[segment_index + 1] if segment_index + 1 < len(onset_values) else len(truth)
        segment_truth = label_values[segment_index]
        correct_rows = sorted(
            (
                row for row in raw_rows
                if onset <= row.time < end and segment_truth in row.node
            ),
            key=lambda row: (row.time, row.emission_index),
        )
        if correct_rows:
            first = correct_rows[0]
            delay = first.time - onset
            boundary_rows.append(BoundaryRow(
                int(stream_index), segment_index, onset, end, segment_truth,
                HIT, True, first.time, delay, False,
            ))
            counts["hit_n"] += 1
            counts["hdel_sum"] += delay
            counts["hdel_n"] += 1
            counts["rdel_sum"] += delay
        else:
            delay = end - onset
            boundary_rows.append(BoundaryRow(
                int(stream_index), segment_index, onset, end, segment_truth,
                RIGHT_CENSORED, False, None, delay, True,
            ))
            counts["miss_n"] += 1
            counts["rdel_sum"] += delay
    counts["bnd_n"] = len(boundary_rows)

    return _make_record(
        counts, flat_ok, tuple(raw_rows), tuple(hierarchy_rows), tuple(boundary_rows)
    )


def pool(records: Sequence[MetricRecord | Mapping[str, Any]]) -> MetricRecord:
    """pool additive counts & derive rates"""
    totals: dict[str, float] = {field: 0 for field in COUNT_FIELDS}
    flat_ok = True
    raw_rows: list[RawEmissionRow] = []
    hierarchy_rows: list[HierarchyEmissionRow] = []
    boundary_rows: list[BoundaryRow] = []

    for stream_index, record in enumerate(records):
        for field in COUNT_FIELDS:
            totals[field] += record[field]
        flat_ok = flat_ok and bool(record["flat_ok"])
        if isinstance(record, MetricRecord):
            raw_rows.extend(
                replace(row, stream_index=stream_index)
                for row in record.raw_emission_rows
            )
            hierarchy_rows.extend(
                replace(row, stream_index=stream_index)
                for row in record.hierarchy_rows
            )
            boundary_rows.extend(
                replace(row, stream_index=stream_index)
                for row in record.boundary_rows
            )

    return _make_record(
        totals, flat_ok, tuple(raw_rows), tuple(hierarchy_rows), tuple(boundary_rows)
    )
