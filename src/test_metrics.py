import math

import numpy as np

from metrics import emission_metrics, pool


def test_empty_and_single_have_same_keys():
    z = np.array([0, 0, 1, 1])
    labs = [0, 1]
    ons = [0, 2]

    a = emission_metrics([], z, labs, ons, K=2)
    b = emission_metrics([(0, 0)], z, labs, ons, K=2)

    assert set(a) == set(b)
    assert a["n_emit"] == 0
    assert math.isnan(a["rep_r"])
    assert math.isnan(a["err_r"])

    assert b["rep_n"] == 0
    assert b["rep_d"] == 0
    assert math.isnan(b["rep_r"])


def test_repeat_is_one_over_two():
    z = np.array([0, 0, 1])
    em = [(0, 0), (1, 0), (2, 1)]

    r = emission_metrics(
        em,
        z,
        labels=[0, 1],
        onsets=[0, 2],
        K=2,
    )

    assert r["rep_n"] == 1
    assert r["rep_d"] == 2
    assert r["rep_r"] == 0.5


def test_perseveration_is_one_over_three():
    z = np.array([0, 0, 0, 1, 1, 1])
    em = [(3, 0), (4, 1), (5, 1)]

    r = emission_metrics(
        em,
        z,
        labels=[0, 1],
        onsets=[0, 3],
        K=2,
    )

    assert r["per_n"] == 1
    assert r["per_d"] == 3
    assert r["per_r"] == 1 / 3


def test_error_is_one_over_four():
    z = np.array([0, 0, 1, 1])
    em = [(0, 0), (1, 0), (2, 0), (3, 1)]

    r = emission_metrics(
        em,
        z,
        labels=[0, 1],
        onsets=[0, 2],
        K=2,
    )

    assert r["err_n"] == 1
    assert r["err_d"] == 4
    assert r["err_r"] == 0.25


def test_boundary_miss_is_retained():
    z = np.array(
        [0] * 5
        + [1] * 5
        + [2] * 5
    )

    r = emission_metrics(
        [(7, 1)],
        z,
        labels=[0, 1, 2],
        onsets=[0, 5, 10],
        K=3,
    )

    assert r["bnd_n"] == 2
    assert r["hit_n"] == 1
    assert r["miss_n"] == 1
    assert r["cov"] == 0.5

    assert r["hdel"] == 2
    assert r["rdel"] == 3.5


def test_hit_to_miss_cannot_improve_result():
    z = np.array(
        [0] * 5
        + [1] * 5
        + [2] * 5
    )

    hit = emission_metrics(
        [(7, 1)],
        z,
        labels=[0, 1, 2],
        onsets=[0, 5, 10],
        K=3,
    )

    miss = emission_metrics(
        [],
        z,
        labels=[0, 1, 2],
        onsets=[0, 5, 10],
        K=3,
    )

    assert miss["cov"] <= hit["cov"]
    assert miss["rdel"] >= hit["rdel"]


def test_nonleaf_keeps_exact_identity():
    z = np.array([1, 1, 2, 2])
    em = [(0, frozenset({0, 1}))]

    r = emission_metrics(
        em,
        z,
        labels=[1, 2],
        onsets=[0, 2],
        K=4,
    )

    assert not r["flat_ok"]
    assert math.isnan(r["err_r"])
    assert math.isnan(r["per_r"])
    assert math.isnan(r["seq_r"])

    assert r["hcov_n"] == 1
    assert r["hcov_d"] == 1
    assert r["hcov_r"] == 1

    assert r["card"] == 2
    assert np.isclose(r["spec"], 2 / 3)


def test_leaf_id_permutation_preserves_tree_metrics():
    z1 = np.array([1, 1, 2, 2])
    z2 = np.array([3, 3, 0, 0])

    a = emission_metrics(
        [(0, frozenset({0, 1}))],
        z1,
        labels=[1, 2],
        onsets=[0, 2],
        K=4,
    )

    b = emission_metrics(
        [(0, frozenset({2, 3}))],
        z2,
        labels=[3, 0],
        onsets=[0, 2],
        K=4,
    )

    assert a["hcov_r"] == b["hcov_r"]
    assert a["card"] == b["card"]
    assert a["spec"] == b["spec"]


def test_pool_uses_counts_not_mean_rates():
    z1 = np.array([0, 0])
    z2 = np.array([0, 1, 2, 3])

    a = emission_metrics(
        [(0, 0), (1, 0)],
        z1,
        labels=[0],
        onsets=[0],
        K=4,
    )

    b = emission_metrics(
        [(0, 0), (1, 1), (2, 2), (3, 3)],
        z2,
        labels=[0, 1, 2, 3],
        onsets=[0, 1, 2, 3],
        K=4,
    )

    r = pool([a, b])

    assert r["rep_n"] == 1
    assert r["rep_d"] == 4
    assert r["rep_r"] == 0.25
