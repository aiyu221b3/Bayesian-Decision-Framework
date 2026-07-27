"""Properties 8-9 for split safety and policy information isolation.

These are red-phase tests for Task 3.9. They intentionally import the future
``protocol`` API inside each test so the suite collects and reports a focused
missing-production-API failure rather than a test-module collection error.
"""

import hashlib
import importlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, seed, strategies as st

from policies import BayesThresh, BocpdThresh, RawThresh
from test_support import HYPOTHESIS_SEED
from tree_belief import TreeBel

ROLES = ("fit", "development", "stress", "test")
SCENARIOS = (
    "no-change",
    "abrupt-change",
    "gradual-coarticulated-change",
    "correlated-evidence",
    "miscalibration",
    "duration-hazard-mismatch",
)
METHODS = ("M0", "M1", "M2", "M3", "MT")
PUBLIC_FIELDS = ("stream_id", "posteriors")
SECRET_FIELDS = (
    "truth", "z", "labels", "durss", "onsets", "bounds",
    "families", "similarity", "generator_settings", "scenario",
)


def _api():
    module = importlib.import_module("protocol")
    return SimpleNamespace(**{
        name: getattr(module, name)
        for name in (
            "ProtocolViolationError", "SimulatorSecretAccessError",
            "StreamManifestEntry", "PolicyView", "ScoringView",
            "validate_manifest", "freeze_study",
        )
    })


def _hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entry(api, *, role, scenario, seed, index, stream_id=None):
    generator_hash = _hash({"K": 4, "scenario": scenario, "seed": seed})
    identifier = stream_id or _hash({
        "role": role,
        "scenario": scenario,
        "seed": seed,
        "index": index,
        "generator_config_hash": generator_hash,
    })
    return api.StreamManifestEntry(
        stream_id=identifier,
        role=role,
        scenario=scenario,
        seed=seed,
        within_seed_index=index,
        policy_visible_field_names=PUBLIC_FIELDS,
        simulator_only_field_names=SECRET_FIELDS,
        generator_config_hash=generator_hash,
        data_hash=_hash({"stream_id": identifier, "frames": index + 3}),
    )


def _manifest(api, seed=0):
    return tuple(
        _entry(
            api,
            role=role,
            scenario=scenario,
            seed=seed,
            index=role_index * len(SCENARIOS) + scenario_index,
        )
        for role_index, role in enumerate(ROLES)
        for scenario_index, scenario in enumerate(SCENARIOS)
    )


def _config(seed=0):
    return {
        "schema_version": "1",
        "seed": seed,
        "threshold": 0.8,
        "roles": ROLES,
        "scenarios": SCENARIOS,
        "methods": METHODS,
    }


def _test_ids(manifest):
    return tuple(row.stream_id for row in manifest if row.role == "test")


def _paired_orders(test_ids):
    return {method: test_ids for method in METHODS}


@st.composite
def manifest_seeds(draw):
    return draw(st.integers(min_value=0, max_value=2**31 - 1))


@st.composite
def invalid_manifest_cases(draw):
    seed_value = draw(manifest_seeds())
    defect = draw(st.sampled_from((
        "duplicate-id", "cross-role-overlap", "incomplete-scenario",
        "unknown-role", "empty-hash",
    )))
    missing = (
        draw(st.sampled_from(SCENARIOS))
        if defect == "incomplete-scenario"
        else None
    )
    return seed_value, defect, missing


def _invalid_manifest(api, case):
    seed_value, defect, missing = case
    manifest = list(_manifest(api, seed_value))
    if defect == "duplicate-id":
        manifest.append(manifest[0])
    elif defect == "cross-role-overlap":
        test_index = next(i for i, row in enumerate(manifest) if row.role == "test")
        manifest[test_index] = replace(
            manifest[test_index], stream_id=manifest[0].stream_id
        )
    elif defect == "incomplete-scenario":
        manifest = [row for row in manifest if row.scenario != missing]
    elif defect == "unknown-role":
        manifest[0] = replace(manifest[0], role="evaluation")
    else:
        manifest[0] = replace(manifest[0], data_hash="")
    return tuple(manifest)


@seed(HYPOTHESIS_SEED)
@given(manifest_seeds())
def test_property_8_generated_valid_manifests_freeze_before_one_paired_pass(seed_value):
    """Property 8. **Validates: Requirements 2.6, 2.7, 3.7**"""
    api = _api()
    manifest = _manifest(api, seed_value)
    validated = api.validate_manifest(manifest, required_scenarios=SCENARIOS)
    study = api.freeze_study(
        validated,
        complete_config=_config(seed_value),
        methods=METHODS,
        scenarios=SCENARIOS,
    )

    assert study.config_hash
    assert study.manifest_hash
    assert study.freeze_timestamp
    expected_ids = _test_ids(manifest)
    observed = []

    def scorer(method, ordered_ids):
        observed.append((method, tuple(ordered_ids)))
        return {"method": method, "n": len(ordered_ids)}

    result = study.final_test_pass(
        _paired_orders(expected_ids),
        scorer,
        current_config=_config(seed_value),
    )

    assert tuple(result) == METHODS
    assert tuple(method for method, _ in observed) == METHODS
    assert all(ids == expected_ids for _, ids in observed)


@seed(HYPOTHESIS_SEED)
@given(invalid_manifest_cases())
def test_property_8_generated_invalid_manifests_abort(case):
    """Property 8. **Validates: Requirements 2.6, 2.7, 3.7**"""
    api = _api()
    manifest = _invalid_manifest(api, case)
    with pytest.raises(api.ProtocolViolationError):
        api.validate_manifest(manifest, required_scenarios=SCENARIOS)


def _frozen_study(api, seed_value=7):
    manifest = _manifest(api, seed_value)
    study = api.freeze_study(
        api.validate_manifest(manifest, required_scenarios=SCENARIOS),
        complete_config=_config(seed_value),
        methods=METHODS,
        scenarios=SCENARIOS,
    )
    return study, manifest


def test_property_8_duplicate_and_cross_role_overlap_abort_before_scoring():
    """Property 8. **Validates: Requirements 2.6, 2.7**"""
    api = _api()
    manifest = list(_manifest(api))
    scorer_calls = []

    for invalid in (
        tuple(manifest + [manifest[0]]),
        tuple(manifest[:-1] + [replace(manifest[-1], stream_id=manifest[0].stream_id)]),
    ):
        with pytest.raises(api.ProtocolViolationError):
            validated = api.validate_manifest(invalid, required_scenarios=SCENARIOS)
            study = api.freeze_study(
                validated, complete_config=_config(), methods=METHODS,
                scenarios=SCENARIOS,
            )
            study.final_test_pass({}, scorer_calls.append, current_config=_config())

    assert scorer_calls == []


def test_property_8_incomplete_scenario_aborts_before_scoring():
    """Property 8. **Validates: Requirement 2.7**"""
    api = _api()
    incomplete = tuple(
        row for row in _manifest(api)
        if row.scenario != "duration-hazard-mismatch"
    )
    scorer_calls = []

    with pytest.raises(api.ProtocolViolationError):
        validated = api.validate_manifest(incomplete, required_scenarios=SCENARIOS)
        study = api.freeze_study(
            validated, complete_config=_config(), methods=METHODS,
            scenarios=SCENARIOS,
        )
        study.final_test_pass({}, scorer_calls.append, current_config=_config())

    assert scorer_calls == []


def test_property_8_order_mismatch_aborts_before_any_method_is_scored():
    """Property 8. **Validates: Requirements 2.6, 3.7**"""
    api = _api()
    study, manifest = _frozen_study(api)
    expected = _test_ids(manifest)
    orders = _paired_orders(expected)
    orders["M3"] = tuple(reversed(expected))
    scorer_calls = []

    with pytest.raises(api.ProtocolViolationError):
        study.final_test_pass(
            orders,
            lambda method, ids: scorer_calls.append((method, tuple(ids))),
            current_config=_config(7),
        )

    assert scorer_calls == []


def test_property_8_post_freeze_mutation_aborts_before_scoring():
    """Property 8. **Validates: Requirements 2.6, 2.7**"""
    api = _api()
    config = _config(11)
    manifest = _manifest(api, 11)
    study = api.freeze_study(
        api.validate_manifest(manifest, required_scenarios=SCENARIOS),
        complete_config=config,
        methods=METHODS,
        scenarios=SCENARIOS,
    )
    assert study.config_hash
    assert study.manifest_hash
    assert study.freeze_timestamp

    config["threshold"] = 0.95
    scorer_calls = []
    with pytest.raises(api.ProtocolViolationError):
        study.final_test_pass(
            _paired_orders(_test_ids(manifest)),
            lambda method, ids: scorer_calls.append((method, tuple(ids))),
            current_config=config,
        )
    assert scorer_calls == []


def test_property_8_second_final_pass_aborts_without_additional_scoring():
    """Property 8. **Validates: Requirements 2.6, 2.7**"""
    api = _api()
    study, manifest = _frozen_study(api, 19)
    orders = _paired_orders(_test_ids(manifest))
    scorer_calls = []

    def scorer(method, ids):
        scorer_calls.append((method, tuple(ids)))
        return len(ids)

    study.final_test_pass(orders, scorer, current_config=_config(19))
    first_pass_calls = tuple(scorer_calls)
    assert len(first_pass_calls) == len(METHODS)

    with pytest.raises(api.ProtocolViolationError):
        study.final_test_pass(orders, scorer, current_config=_config(19))

    assert tuple(scorer_calls) == first_pass_calls


@st.composite
def isolation_cases(draw):
    length = draw(st.integers(min_value=2, max_value=16))
    weights = draw(st.lists(
        st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=4,
            max_size=4,
        ),
        min_size=length,
        max_size=length,
    ))
    posteriors = np.asarray(weights, dtype=float)
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    offset = draw(st.integers(min_value=0, max_value=3))

    hidden = []
    for index, scenario in enumerate(SCENARIOS):
        truth = tuple(
            int((frame + offset + index) % 4)
            for frame in range(length)
        )
        hidden.append({
            "scenario": scenario,
            "truth": truth,
            "families": tuple(np.roll(np.arange(4) % 2, index % 4)),
            "settings": {
                "hazard": (index + 1) / 100,
                "temperature": 0.5 + index / 10,
                "coarticulation": index,
            },
        })
    return posteriors, tuple(hidden)


def _policy_outputs(view):
    similarity = np.asarray([
        [1.0, 0.8, 0.1, 0.1],
        [0.8, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0],
    ])
    policies = {
        "M0": RawThresh(4, theta=0.8),
        "M1": BayesThresh(4, rho=0.0, tau=0.8),
        "M2": BayesThresh(4, rho=0.05, tau=0.8),
        "M3": BocpdThresh(4, tau=0.8, cp_thr=0.1),
        "MT": TreeBel(4, similarity, tau=0.8, min_sp=0.25),
    }
    return {
        method: tuple(policy.run(view))
        for method, policy in policies.items()
    }


@seed(HYPOTHESIS_SEED)
@given(isolation_cases())
def test_property_9_identical_visible_traces_ignore_all_hidden_variants(case):
    """Property 9. **Validates: Requirements 2.7, 3.2, 3.7, 3.8**"""
    api = _api()
    posteriors, hidden_variants = case
    outputs = []

    for hidden in hidden_variants:
        policy_view = api.PolicyView("held-constant", posteriors)
        scoring_view = api.ScoringView(
            stream_id="held-constant",
            truth=hidden["truth"],
            labels=(hidden["truth"][0],),
            onsets=(0,),
            bounds=(),
            hierarchy_truth=tuple(
                frozenset((label,)) for label in hidden["truth"]
            ),
        )
        assert scoring_view.stream_id == policy_view.stream_id
        assert np.array_equal(policy_view.posteriors, posteriors)
        assert set(dir(policy_view)).isdisjoint(SECRET_FIELDS)
        outputs.append(_policy_outputs(policy_view))

    assert tuple(hidden["scenario"] for hidden in hidden_variants) == SCENARIOS
    assert all(output == outputs[0] for output in outputs[1:])


@pytest.mark.parametrize("secret", SECRET_FIELDS)
@pytest.mark.parametrize("access_mode", ("attribute", "item", "get"))
def test_property_9_every_simulator_secret_access_raises_protocol_violation(
    secret, access_mode
):
    """Property 9. **Validates: Requirements 2.7, 3.8**"""
    api = _api()
    view = api.PolicyView("stream", np.asarray([[0.8, 0.2], [0.1, 0.9]]))

    def access():
        if access_mode == "attribute":
            return getattr(view, secret)
        if access_mode == "item":
            return view[secret]
        return view.get(secret)

    with pytest.raises(api.SimulatorSecretAccessError):
        access()


@seed(HYPOTHESIS_SEED)
@given(
    st.sampled_from(SECRET_FIELDS),
    st.sampled_from(("attribute", "item", "get")),
)
def test_property_9_generated_secret_access_attempts_always_abort(secret, access_mode):
    """Property 9. **Validates: Requirements 2.7, 3.8**"""
    api = _api()
    view = api.PolicyView("stream", np.asarray([[0.5, 0.5]]))

    with pytest.raises(api.SimulatorSecretAccessError):
        if access_mode == "attribute":
            getattr(view, secret)
        elif access_mode == "item":
            view[secret]
        else:
            view.get(secret)


def test_property_9_policy_view_is_public_only_read_only_and_defensively_copied():
    """Property 9. **Validates: Requirements 2.7, 3.8**"""
    api = _api()
    source = np.asarray([[0.75, 0.25], [0.2, 0.8]])
    view = api.PolicyView("stream", source)
    expected = source.copy()

    source[0, 0] = 0.0
    assert view.stream_id == "stream"
    assert np.array_equal(view.posteriors, expected)
    assert not view.posteriors.flags.writeable
    assert not (set(dir(view)) & set(SECRET_FIELDS))

    with pytest.raises((ValueError, TypeError, AttributeError)):
        view.posteriors[0, 0] = 1.0
    with pytest.raises((TypeError, AttributeError)):
        view.stream_id = "mutated"
