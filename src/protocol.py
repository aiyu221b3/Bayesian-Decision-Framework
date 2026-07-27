"""split protocol & sim boundaries"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import numpy as np


ROLES = ("fit", "development", "stress", "test")
SCENARIOS = (
    "no-change",
    "abrupt-change",
    "gradual-coarticulated-change",
    "correlated-evidence",
    "miscalibration",
    "duration-hazard-mismatch",
)
POLICY_FIELDS = ("stream_id", "posteriors")
SIMULATOR_SECRET_FIELDS = frozenset({
    "truth", "z", "labels", "durss", "onsets", "bounds",
    "families", "family", "similarity", "latent_similarity",
    "generator_settings", "settings", "simulator_settings", "scenario",
    "hierarchy_truth", "scoring", "scoring_view", "manifest_entry",
})


class ProtocolViolationError(RuntimeError):
    """study invariant violation"""


class SimulatorSecretAccessError(ProtocolViolationError):
    """sim secret access error"""


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"dtype": array.dtype.str, "shape": list(array.shape),
                "data": array.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = (_canonical(item) for item in value)
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported value in protocol hash: {type(value).__name__}")


def content_hash(payload: Any) -> str:
    """sha-256 digest of content"""
    payload_json = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def derive_stream_id(role: str, scenario: str, seed: int,
                     within_seed_index: int, generator_config_hash: str) -> str:
    """stream id from gen metadata"""
    return content_hash({
        "role": role,
        "scenario": scenario,
        "seed": int(seed),
        "index": int(within_seed_index),
        "generator_config_hash": generator_config_hash,
    })


def _hash_is_valid(value: str) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


@dataclass(frozen=True, slots=True)
class StreamManifestEntry:
    stream_id: str
    role: str
    scenario: str
    seed: int
    within_seed_index: int
    policy_visible_field_names: tuple[str, ...]
    simulator_only_field_names: tuple[str, ...]
    generator_config_hash: str
    data_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "within_seed_index", int(self.within_seed_index))
        object.__setattr__(self, "policy_visible_field_names",
                           tuple(self.policy_visible_field_names))
        object.__setattr__(self, "simulator_only_field_names",
                           tuple(self.simulator_only_field_names))


class PolicyView:
    """read-only array policy projection with secret guards"""

    __slots__ = ("_stream_id", "_posteriors")

    def __init__(self, stream_id: str, posteriors: Any):
        array = np.asarray(posteriors)
        if array.ndim != 2:
            raise ValueError("posteriors must be a two-dimensional frame matrix")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("posteriors must be finite and non-negative")
        if not np.allclose(array.sum(axis=1), 1.0):
            raise ValueError("each posterior row must sum to one")
        owned = np.ascontiguousarray(array).copy()
        immutable = np.frombuffer(owned.tobytes(), dtype=owned.dtype).reshape(owned.shape)
        object.__setattr__(self, "_stream_id", str(stream_id))
        object.__setattr__(self, "_posteriors", immutable)

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def posteriors(self) -> np.ndarray:
        return self._posteriors

    def __len__(self) -> int:
        return len(self._posteriors)

    def __iter__(self):
        return iter(self._posteriors)

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self._posteriors, dtype=dtype)
        return array.copy() if copy else array

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in SIMULATOR_SECRET_FIELDS:
                raise SimulatorSecretAccessError(
                    f"policy attempted simulator-secret access: {key}"
                )
            if key == "stream_id":
                return self._stream_id
            if key in ("posteriors", "p"):
                return self._posteriors
            raise KeyError(key)
        return self._posteriors[key]

    def get(self, key: str, default=None):
        if key in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {key}"
            )
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, name: str):
        if name in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {name}"
            )
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PolicyView is immutable")


class PolicyVisibleDict(dict):
    """dict projection of policy visible fields"""

    def __getitem__(self, key):
        if key in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {key}"
            )
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {key}"
            )
        return super().get(key, default)


@dataclass(frozen=True, slots=True)
class ScoringView:
    """immutable sim truth projection for scorers"""

    stream_id: str
    truth: tuple[int, ...]
    labels: tuple[int, ...]
    onsets: tuple[int, ...]
    bounds: tuple[int, ...]
    hierarchy_truth: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", str(self.stream_id))
        for field in ("truth", "labels", "onsets", "bounds"):
            object.__setattr__(self, field,
                               tuple(int(value) for value in getattr(self, field)))
        object.__setattr__(self, "hierarchy_truth", tuple(
            frozenset(int(value) for value in node)
            for node in self.hierarchy_truth
        ))
        if len(self.hierarchy_truth) != len(self.truth):
            raise ValueError("hierarchy_truth must have one node per truth frame")


def make_manifest_entry(*, role: str, scenario: str, seed: int,
                        within_seed_index: int, generator_config: Any,
                        policy_view: PolicyView, scoring_view: ScoringView,
                        simulator_only_field_names: Iterable[str] =
                        SIMULATOR_SECRET_FIELDS) -> StreamManifestEntry:
    """build manifest row from stream content"""
    generator_hash = content_hash(generator_config)
    stream_id = derive_stream_id(role, scenario, seed, within_seed_index,
                                 generator_hash)
    if policy_view.stream_id != stream_id or scoring_view.stream_id != stream_id:
        raise ProtocolViolationError("view stream IDs do not match derived stream ID")
    data_hash = content_hash({
        "stream_id": stream_id,
        "posteriors": policy_view.posteriors,
        "truth": scoring_view.truth,
        "labels": scoring_view.labels,
        "onsets": scoring_view.onsets,
        "bounds": scoring_view.bounds,
        "hierarchy_truth": scoring_view.hierarchy_truth,
    })
    return StreamManifestEntry(
        stream_id=stream_id,
        role=role,
        scenario=scenario,
        seed=seed,
        within_seed_index=within_seed_index,
        policy_visible_field_names=POLICY_FIELDS,
        simulator_only_field_names=tuple(sorted(simulator_only_field_names)),
        generator_config_hash=generator_hash,
        data_hash=data_hash,
    )


def validate_manifest(manifest: Iterable[StreamManifestEntry], *,
                      required_scenarios: Iterable[str] = SCENARIOS,
                      required_roles: Iterable[str] = ROLES
                      ) -> tuple[StreamManifestEntry, ...]:
    """validate identity, role disjointness, hashes, scenarios"""
    rows = tuple(manifest)
    roles = tuple(required_roles)
    scenarios = tuple(required_scenarios)
    if not rows:
        raise ProtocolViolationError("manifest cannot be empty")
    if len(set(roles)) != len(roles) or not set(roles).issubset(ROLES):
        raise ProtocolViolationError("required roles must be known and unique")
    if len(set(scenarios)) != len(scenarios) or not set(scenarios).issubset(SCENARIOS):
        raise ProtocolViolationError("required scenarios must be known and unique")

    seen: dict[str, str] = {}
    coverage = {role: set() for role in roles}
    for row in rows:
        if not isinstance(row, StreamManifestEntry):
            raise ProtocolViolationError("manifest contains a non-entry value")
        if row.role not in roles:
            raise ProtocolViolationError(f"unknown or undeclared role: {row.role}")
        if row.scenario not in scenarios:
            raise ProtocolViolationError(f"unknown or undeclared scenario: {row.scenario}")
        if row.within_seed_index < 0:
            raise ProtocolViolationError("within-seed index cannot be negative")
        if not _hash_is_valid(row.generator_config_hash) or not _hash_is_valid(row.data_hash):
            raise ProtocolViolationError("manifest hashes must be nonempty SHA-256 values")
        expected_id = derive_stream_id(row.role, row.scenario, row.seed,
                                       row.within_seed_index,
                                       row.generator_config_hash)
        if row.stream_id != expected_id:
            raise ProtocolViolationError("stream ID does not match frozen content metadata")
        if row.stream_id in seen:
            raise ProtocolViolationError(
                f"duplicate or cross-role stream ID overlap: {row.stream_id}"
            )
        if tuple(row.policy_visible_field_names) != POLICY_FIELDS:
            raise ProtocolViolationError("policy-visible fields must be ID and posteriors only")
        if set(row.simulator_only_field_names).isdisjoint(SIMULATOR_SECRET_FIELDS):
            raise ProtocolViolationError("manifest does not declare simulator-only fields")
        seen[row.stream_id] = row.role
        coverage[row.role].add(row.scenario)

    required = set(scenarios)
    for role, observed in coverage.items():
        missing = required - observed
        if missing:
            raise ProtocolViolationError(
                f"incomplete scenario coverage for {role}: {sorted(missing)}"
            )
    return rows


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        owned = np.ascontiguousarray(array).copy()
        immutable = np.frombuffer(owned.tobytes(), dtype=owned.dtype).reshape(owned.shape)
        object.__setattr__(self, "_stream_id", str(stream_id))
        object.__setattr__(self, "_posteriors", immutable)

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def posteriors(self) -> np.ndarray:
        return self._posteriors

    def __len__(self) -> int:
        return len(self._posteriors)

    def __iter__(self):
        return iter(self._posteriors)

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self._posteriors, dtype=dtype)
        return array.copy() if copy else array

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in SIMULATOR_SECRET_FIELDS:
                raise SimulatorSecretAccessError(
                    f"policy attempted simulator-secret access: {key}"
                )
            if key == "stream_id":
                return self._stream_id
            if key in ("posteriors", "p"):
                return self._posteriors
            raise KeyError(key)
        return self._posteriors[key]

    def get(self, key: str, default=None):
        if key in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {key}"
            )
        try:
            return self[key]
        except KeyError:
            return default

    def __getattr__(self, name: str):
        if name in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {name}"
            )
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PolicyView is immutable")


class PolicyVisibleDict(dict):
    """dict projection of policy visible fields"""

    def __getitem__(self, key):
        if key in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {key}"
            )
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in SIMULATOR_SECRET_FIELDS:
            raise SimulatorSecretAccessError(
                f"policy attempted simulator-secret access: {key}"
            )
        return super().get(key, default)


@dataclass(frozen=True, slots=True)
class ScoringView:
    """immutable sim truth projection for scorers"""

    stream_id: str
    truth: tuple[int, ...]
    labels: tuple[int, ...]
    onsets: tuple[int, ...]
    bounds: tuple[int, ...]
    hierarchy_truth: tuple[frozenset[int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", str(self.stream_id))
        for field in ("truth", "labels", "onsets", "bounds"):
            object.__setattr__(self, field,
                               tuple(int(value) for value in getattr(self, field)))
        object.__setattr__(self, "hierarchy_truth", tuple(
            frozenset(int(value) for value in node)
            for node in self.hierarchy_truth
        ))
        if len(self.hierarchy_truth) != len(self.truth):
            raise ValueError("hierarchy_truth must have one node per truth frame")


def make_manifest_entry(*, role: str, scenario: str, seed: int,
                        within_seed_index: int, generator_config: Any,
                        policy_view: PolicyView, scoring_view: ScoringView,
                        simulator_only_field_names: Iterable[str] =
                        SIMULATOR_SECRET_FIELDS) -> StreamManifestEntry:
    """build manifest row from stream content"""
    generator_hash = content_hash(generator_config)
    stream_id = derive_stream_id(role, scenario, seed, within_seed_index,
                                 generator_hash)
    if policy_view.stream_id != stream_id or scoring_view.stream_id != stream_id:
        raise ProtocolViolationError("view stream IDs do not match derived stream ID")
    data_hash = content_hash({
        "stream_id": stream_id,
        "posteriors": policy_view.posteriors,
        "truth": scoring_view.truth,
        "labels": scoring_view.labels,
        "onsets": scoring_view.onsets,
        "bounds": scoring_view.bounds,
        "hierarchy_truth": scoring_view.hierarchy_truth,
    })
    return StreamManifestEntry(
        stream_id=stream_id,
        role=role,
        scenario=scenario,
        seed=seed,
        within_seed_index=within_seed_index,
        policy_visible_field_names=POLICY_FIELDS,
        simulator_only_field_names=tuple(sorted(simulator_only_field_names)),
        generator_config_hash=generator_hash,
        data_hash=data_hash,
    )


def validate_manifest(manifest: Iterable[StreamManifestEntry], *,
                      required_scenarios: Iterable[str] = SCENARIOS,
                      required_roles: Iterable[str] = ROLES
                      ) -> tuple[StreamManifestEntry, ...]:
    """validate identity, role disjointness, hashes, scenarios"""
    rows = tuple(manifest)
    roles = tuple(required_roles)
    scenarios = tuple(required_scenarios)
    if not rows:
        raise ProtocolViolationError("manifest cannot be empty")
    if len(set(roles)) != len(roles) or not set(roles).issubset(ROLES):
        raise ProtocolViolationError("required roles must be known and unique")
    if len(set(scenarios)) != len(scenarios) or not set(scenarios).issubset(SCENARIOS):
        raise ProtocolViolationError("required scenarios must be known and unique")

    seen: dict[str, str] = {}
    coverage = {role: set() for role in roles}
    for row in rows:
        if not isinstance(row, StreamManifestEntry):
            raise ProtocolViolationError("manifest contains a non-entry value")
        if row.role not in roles:
            raise ProtocolViolationError(f"unknown or undeclared role: {row.role}")
        if row.scenario not in scenarios:
            raise ProtocolViolationError(f"unknown or undeclared scenario: {row.scenario}")
        if row.within_seed_index < 0:
            raise ProtocolViolationError("within-seed index cannot be negative")
        if not _hash_is_valid(row.generator_config_hash) or not _hash_is_valid(row.data_hash):
            raise ProtocolViolationError("manifest hashes must be nonempty SHA-256 values")
        expected_id = derive_stream_id(row.role, row.scenario, row.seed,
                                       row.within_seed_index,
                                       row.generator_config_hash)
        if row.stream_id != expected_id:
            raise ProtocolViolationError("stream ID does not match frozen content metadata")
        if row.stream_id in seen:
            raise ProtocolViolationError(
                f"duplicate or cross-role stream ID overlap: {row.stream_id}"
            )
        if tuple(row.policy_visible_field_names) != POLICY_FIELDS:
            raise ProtocolViolationError("policy-visible fields must be ID and posteriors only")
        if set(row.simulator_only_field_names).isdisjoint(SIMULATOR_SECRET_FIELDS):
            raise ProtocolViolationError("manifest does not declare simulator-only fields")
        seen[row.stream_id] = row.role
        coverage[row.role].add(row.scenario)

    required = set(scenarios)
    for role, observed in coverage.items():
        missing = required - observed
        if missing:
            raise ProtocolViolationError(
                f"incomplete scenario coverage for {role}: {sorted(missing)}"
            )
    return rows


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    return value


class FrozenStudy:
    """frozen protocol state"""

    __slots__ = (
        "_manifest", "_complete_config", "_methods", "_scenarios",
        "_role_ids", "_final_pass_used", "config_hash", "manifest_hash",
        "freeze_timestamp", "schema_version",
    )

    def __init__(self, manifest: tuple[StreamManifestEntry, ...],
                 complete_config: Mapping[str, Any], methods: Iterable[str],
                 scenarios: Iterable[str], schema_version: str = "1"):
        methods_tuple = tuple(str(method) for method in methods)
        scenarios_tuple = tuple(str(scenario) for scenario in scenarios)
        if not methods_tuple or len(set(methods_tuple)) != len(methods_tuple):
            raise ProtocolViolationError("methods must be nonempty and unique")
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_complete_config", _deep_freeze(complete_config))
        object.__setattr__(self, "_methods", methods_tuple)
        object.__setattr__(self, "_scenarios", scenarios_tuple)
        object.__setattr__(self, "_role_ids", MappingProxyType({
            role: tuple(row.stream_id for row in manifest if row.role == role)
            for role in ROLES
        }))
        object.__setattr__(self, "config_hash", content_hash(complete_config))
        object.__setattr__(self, "manifest_hash", content_hash(manifest))
        object.__setattr__(self, "freeze_timestamp",
                           datetime.now(timezone.utc).isoformat())
        object.__setattr__(self, "schema_version", str(schema_version))
        object.__setattr__(self, "_final_pass_used", False)

    @property
    def methods(self) -> tuple[str, ...]:
        return self._methods

    @property
    def scenarios(self) -> tuple[str, ...]:
        return self._scenarios

    @property
    def complete_config(self) -> Mapping[str, Any]:
        return self._complete_config

    def ids_for_role(self, role: str) -> tuple[str, ...]:
        if role == "test":
            raise ProtocolViolationError(
                "test IDs are available only through the one final test pass"
            )
        if role not in ("fit", "development", "stress"):
            raise ProtocolViolationError(f"unknown role: {role}")
        return self._role_ids[role]

    def require_role(self, entries: Iterable[StreamManifestEntry], role: str,
                     purpose: str) -> tuple[StreamManifestEntry, ...]:
        """guard fit/dev/stress apis against role misuse"""
        allowed = {
            "fit": ("confusion", "taxonomy", "dp", "tables"),
            "development": ("scalar", "threshold", "selection"),
            "stress": ("mechanism", "frozen", "check"),
        }
        if role not in allowed or not any(token in purpose.lower()
                                          for token in allowed[role]):
            raise ProtocolViolationError(f"{purpose} is not permitted for role {role}")
        rows = tuple(entries)
        if any(row.role != role for row in rows):
            raise ProtocolViolationError(f"{purpose} requires only {role} streams")
        return rows

    def final_test_pass(self, ordered_ids_by_method: Mapping[str, Iterable[str]],
                        scorer: Callable[[str, tuple[str, ...]], Any], *,
                        current_config: Mapping[str, Any]) -> Mapping[str, Any]:
        """validate final-pass invariants before scoring"""
        if self._final_pass_used:
            raise ProtocolViolationError("the final test pass has already been consumed")
        if content_hash(current_config) != self.config_hash:
            raise ProtocolViolationError("configuration changed after study freeze")
        if content_hash(self._manifest) != self.manifest_hash:
            raise ProtocolViolationError("manifest changed after study freeze")
        if set(ordered_ids_by_method) != set(self._methods):
            raise ProtocolViolationError("every frozen method must provide test IDs exactly once")

        expected = self._role_ids["test"]
        checked: dict[str, tuple[str, ...]] = {}
        for method in self._methods:
            supplied = tuple(ordered_ids_by_method[method])
            if supplied != expected:
                raise ProtocolViolationError(
                    f"ordered test-ID mismatch for method {method}"
                )
            checked[method] = supplied

        object.__setattr__(self, "_final_pass_used", True)
        results = {method: scorer(method, checked[method]) for method in self._methods}
        return MappingProxyType(results)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("FrozenStudy is immutable")


def freeze_study(manifest: Iterable[StreamManifestEntry], *,
                 complete_config: Mapping[str, Any], methods: Iterable[str],
                 scenarios: Iterable[str] = SCENARIOS,
                 schema_version: str = "1") -> FrozenStudy:
    """validate and freeze study decisions before test access"""
    scenario_tuple = tuple(scenarios)
    rows = validate_manifest(manifest, required_scenarios=scenario_tuple)
    return FrozenStudy(rows, complete_config, methods, scenario_tuple, schema_version)
