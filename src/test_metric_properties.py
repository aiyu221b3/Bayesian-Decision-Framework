"""Properties 5-7 for stable, failure-aware flat and hierarchy metrics.

The hand-counted acceptance fixtures live in ``test_metrics.py``; this module
extends them over deterministic bounded Hypothesis generators.
"""

import math

import numpy as np
from hypothesis import given, seed, strategies as st

from metrics import emission_metrics, node as metric_node, pool
from test_support import HYPOTHESIS_SEED, NUMERIC_TOLERANCE
from tree_belief import specificity

ATOL = NUMERIC_TOLERANCE

COUNT_FIELDS = (
    "n_emit", "rep_n", "rep_d", "err_n", "err_d", "per_n", "per_d",
    "seq_n", "seq_d", "bnd_n", "hit_n", "miss_n", "rdel_sum",
    "hdel_sum", "hdel_n", "hcov_n", "hcov_d", "card_sum", "card_n",
    "spec_sum", "spec_n",
)
RATE_FIELDS = {
    "rep_r": ("rep_n", "rep_d"),
    "err_r": ("err_n", "err_d"),
    "per_r": ("per_n", "per_d"),
    "seq_r": ("seq_n", "seq_d"),
    "cov": ("hit_n", "bnd_n"),
    "rdel": ("rdel_sum", "bnd_n"),
    "hdel": ("hdel_sum", "hdel_n"),
    "hcov_r": ("hcov_n", "hcov_d"),
    "card": ("card_sum", "card_n"),
    "spec": ("spec_sum", "spec_n"),
}
SCHEMA = frozenset(COUNT_FIELDS) | frozenset(RATE_FIELDS) | {"flat_ok"}


def _labels(draw, classes, segments):
    labels = [draw(st.integers(min_value=0, max_value=classes - 1))]
    for _ in range(1, segments):
        labels.append(draw(st.sampled_from(
            tuple(x for x in range(classes) if x != labels[-1])
        )))
    return labels


def _stream(labels, durss):
    z = np.concatenate([
        np.full(duration, label, dtype=int)
        for label, duration in zip(labels, durss)
    ])
    onsets = [0] + list(np.cumsum(durss)[:-1])
    return z, onsets


@st.composite
def flat_cases(draw):
    classes = draw(st.integers(min_value=2, max_value=6))
    segments = draw(st.integers(min_value=1, max_value=5))
    labels = _labels(draw, classes, segments)
    durss = draw(st.lists(
        st.integers(min_value=1, max_value=6),
        min_size=segments,
        max_size=segments,
    ))
    z, onsets = _stream(labels, durss)
    count = draw(st.integers(min_value=0, max_value=min(len(z), 12)))
    times = sorted(draw(st.lists(
        st.integers(min_value=0, max_value=len(z) - 1),
        min_size=count,
        max_size=count,
        unique=True,
    )))
    outputs = draw(st.lists(
        st.integers(min_value=0, max_value=classes - 1),
        min_size=count,
        max_size=count,
    ))
    return [(t, y) for t, y in zip(times, outputs)], z, labels, onsets, classes


def _assert_rate_identities(record):
    for rate_key, (num_key, den_key) in RATE_FIELDS.items():
        denominator = record[den_key]
        if denominator == 0:
            assert record[num_key] == 0
            assert math.isnan(record[rate_key])
        else:
            assert np.isclose(
                record[rate_key],
                record[num_key] / denominator,
                atol=ATOL,
            )


@seed(HYPOTHESIS_SEED)
@given(flat_cases())
def test_property_5_generated_flat_metrics_have_stable_additive_counts_and_rates(case):
    """Property 5. **Validates: Requirements 2.3, 3.5**"""
    emissions, z, labels, onsets, classes = case
    record = emission_metrics(emissions, z, labels, onsets, K=classes)

    assert frozenset(record) == SCHEMA
    assert record["flat_ok"] is True
    assert record["n_emit"] == len(emissions)
    assert record["rep_d"] == max(len(emissions) - 1, 0)
    assert record["rep_n"] == sum(
        left[1] == right[1]
        for left, right in zip(emissions, emissions[1:])
    )
    assert record["err_d"] == len(emissions)
    assert record["err_n"] == sum(y != int(z[t]) for t, y in emissions)
    _assert_rate_identities(record)


@seed(HYPOTHESIS_SEED)
@given(st.lists(flat_cases(), min_size=1, max_size=5))
def test_property_5_pool_sums_opportunities_before_deriving_rates(cases):
    """Property 5. **Validates: Requirements 2.3, 3.5**"""
    rows = [
        emission_metrics(emissions, z, labels, onsets, K=classes)
        for emissions, z, labels, onsets, classes in cases
    ]
    aggregate = pool(rows)

    assert frozenset(aggregate) == SCHEMA
    for field in COUNT_FIELDS:
        assert np.isclose(
            aggregate[field],
            sum(row[field] for row in rows),
            atol=ATOL,
        )
    assert aggregate["flat_ok"] is True
    _assert_rate_identities(aggregate)


@st.composite
def segmented_cases(draw):
    classes = draw(st.integers(min_value=2, max_value=6))
    segments = draw(st.integers(min_value=1, max_value=6))
    labels = _labels(draw, classes, segments)
    durss = draw(st.lists(
        st.integers(min_value=1, max_value=8),
        min_size=segments,
        max_size=segments,
    ))
    z, onsets = _stream(labels, durss)

    emissions = []
    hit_rows = []
    restricted = []
    hit_delays = []
    for segment in range(1, segments):
        is_hit = draw(st.booleans())
        if is_hit:
            delay = draw(st.integers(min_value=0, max_value=durss[segment] - 1))
            row = (onsets[segment] + delay, labels[segment])
            emissions.append(row)
            hit_rows.append(row)
            hit_delays.append(delay)
            restricted.append(delay)
        else:
            restricted.append(durss[segment])

    return (
        emissions, z, labels, onsets, classes, hit_rows,
        restricted, hit_delays,
    )


@seed(HYPOTHESIS_SEED)
@given(segmented_cases())
def test_property_6_each_boundary_is_exactly_one_hit_or_right_censored_miss(case):
    """Property 6. **Validates: Requirement 2.4**"""
    emissions, z, labels, onsets, classes, hits, restricted, hit_delays = case
    record = emission_metrics(emissions, z, labels, onsets, K=classes)
    eligible = len(onsets) - 1

    assert record["bnd_n"] == eligible
    assert record["hit_n"] == len(hits)
    assert record["miss_n"] == eligible - len(hits)
    assert record["hit_n"] + record["miss_n"] == record["bnd_n"]
    assert np.isclose(record["rdel_sum"], sum(restricted), atol=ATOL)
    assert np.isclose(record["hdel_sum"], sum(hit_delays), atol=ATOL)
    assert record["hdel_n"] == len(hit_delays)
    _assert_rate_identities(record)


@seed(HYPOTHESIS_SEED)
@given(segmented_cases())
def test_property_6_converting_a_hit_to_a_miss_cannot_improve_primary_metrics(case):
    """Property 6. **Validates: Requirement 2.4**"""
    emissions, z, labels, onsets, classes, hits, _, _ = case
    if not hits:
        return

    before = emission_metrics(emissions, z, labels, onsets, K=classes)
    removed = hits[0]
    after_emissions = list(emissions)
    after_emissions.remove(removed)
    after = emission_metrics(after_emissions, z, labels, onsets, K=classes)

    assert after["bnd_n"] == before["bnd_n"]
    assert after["hit_n"] == before["hit_n"] - 1
    assert after["miss_n"] == before["miss_n"] + 1
    assert after["cov"] <= before["cov"]
    assert after["rdel"] >= before["rdel"]


@st.composite
def hierarchy_cases(draw):
    classes = draw(st.integers(min_value=2, max_value=8))
    order = draw(st.permutations(tuple(range(classes))))
    tree = [frozenset((leaf,)) for leaf in order]
    tree.extend(frozenset(order[:width]) for width in range(2, classes + 1))
    selected = draw(st.sampled_from(tuple(tree)))

    segments = draw(st.integers(min_value=1, max_value=4))
    labels = _labels(draw, classes, segments)
    durss = draw(st.lists(
        st.integers(min_value=1, max_value=5),
        min_size=segments,
        max_size=segments,
    ))
    z, onsets = _stream(labels, durss)
    time = draw(st.integers(min_value=0, max_value=len(z) - 1))

    permutation = draw(st.permutations(tuple(range(classes))))
    mapping = {old: permutation[old] for old in range(classes)}
    mapped_tree = [frozenset(mapping[x] for x in nd) for nd in tree]
    mapped_node = frozenset(mapping[x] for x in selected)
    mapped_z = np.asarray([mapping[int(x)] for x in z], dtype=int)
    mapped_labels = [mapping[x] for x in labels]

    return (
        classes, tree, selected, z, labels, onsets, time,
        mapping, mapped_tree, mapped_node, mapped_z, mapped_labels,
    )


def _same_float(left, right):
    if math.isnan(left) and math.isnan(right):
        return True
    return bool(np.isclose(left, right, atol=ATOL))


@seed(HYPOTHESIS_SEED)
@given(hierarchy_cases())
def test_property_7_structure_preserving_leaf_bijections_are_equivariant(case):
    """Property 7. **Validates: Requirements 2.5, 3.6**"""
    (
        classes, tree, selected, z, labels, onsets, time,
        mapping, mapped_tree, mapped_node, mapped_z, mapped_labels,
    ) = case
    original = emission_metrics(
        [(time, selected)], z, labels, onsets, K=classes
    )
    transformed = emission_metrics(
        [(time, mapped_node)], mapped_z, mapped_labels, onsets, K=classes
    )

    assert mapped_tree == [
        frozenset(mapping[x] for x in nd)
        for nd in tree
    ]
    assert metric_node(mapped_node) == frozenset(mapping[x] for x in selected)
    assert original["flat_ok"] == transformed["flat_ok"]

    invariant_fields = (
        "hcov_n", "hcov_d", "hcov_r", "card_sum", "card_n", "card",
        "spec_sum", "spec_n", "spec", "bnd_n", "hit_n", "miss_n",
        "rdel_sum", "rdel", "hdel_sum", "hdel_n", "hdel",
    )
    for field in invariant_fields:
        assert _same_float(float(original[field]), float(transformed[field]))

    if len(selected) > 1:
        assert not original["flat_ok"]
        assert not transformed["flat_ok"]
        for field in ("err_r", "per_r", "seq_r"):
            assert math.isnan(original[field])
            assert math.isnan(transformed[field])
        assert metric_node(mapped_node) != frozenset((min(mapped_node),))


@seed(HYPOTHESIS_SEED)
@given(hierarchy_cases())
def test_property_7_specificity_is_strictly_monotone_down_ancestor_chains(case):
    """Property 7. **Validates: Requirements 2.5, 3.6**"""
    classes, tree, _, z, labels, onsets, time, *_ = case

    for child in tree:
        for ancestor in tree:
            if child < ancestor:
                assert specificity(child, classes) > specificity(ancestor, classes)
                child_record = emission_metrics(
                    [(time, child)], z, labels, onsets, K=classes
                )
                ancestor_record = emission_metrics(
                    [(time, ancestor)], z, labels, onsets, K=classes
                )
                assert child_record["spec"] > ancestor_record["spec"]


def test_property_7_exact_leaf_family_root_and_disjoint_family_fixtures():
    """Property 7. **Validates: Requirements 2.5, 3.6**"""
    z = np.asarray([0, 0, 0], dtype=int)
    cases = (
        (frozenset({0}), 1, 1.0),
        (frozenset({0, 1}), 1, 2 / 3),
        (frozenset({0, 1, 2, 3}), 1, 0.0),
        (frozenset({1, 2}), 0, 2 / 3),
    )
    for identity, covered, expected_specificity in cases:
        record = emission_metrics([(1, identity)], z, [0], [0], K=4)
        assert record["hcov_n"] == covered
        assert record["hcov_d"] == 1
        assert record["card"] == len(identity)
        assert np.isclose(record["spec"], expected_specificity, atol=ATOL)
        if len(identity) > 1:
            assert not record["flat_ok"]
            assert all(math.isnan(record[key]) for key in ("err_r", "per_r", "seq_r"))
