import numpy as np
from evidence import (
    Simcon,
    _label_seq,
    _mean_logits,
    build_sim,
    concat_stream,
    generate_stream,
)
from test_support import NUMERIC_TOLERANCE


def test_similarity_matrix_symmetric_and_unit_diagonal():
    cfg = Simcon()
    rng = np.random.default_rng(0)
    S, fam = build_sim(cfg, rng)
    assert np.allclose(S, S.T), "S should be symmetric (S[i,j] == S[j,i] for all pairs)"
    assert np.allclose(np.diag(S), 1.0), "every diagonal entry should be exactly 1.0"


def test_no_consecutive_repeated_labels():
    cfg = Simcon()
    rng = np.random.default_rng(0)
    for _ in range(50):  # repeat many times since the sequence is randomized
        z, labels, durs, onsets, bounds = _label_seq(cfg, 6, rng)
        for a, b in zip(labels, labels[1:]):
            assert a != b, f"found two identical signs back to back: {labels}"


def test_frame_count_matches_duration_sum():
    cfg = Simcon()
    rng = np.random.default_rng(0)
    z, labels, durs, onsets, bounds = _label_seq(cfg, 4, rng)
    assert len(z) == sum(durs)


def test_probabilities_sum_to_one():
    cfg = Simcon()
    rng = np.random.default_rng(0)
    out = concat_stream(cfg, rng, n_signs=4)
    row_sums = out["p"].sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=NUMERIC_TOLERANCE), \
        "every frame's probabilities must sum to 1"


def test_blend_handoff_has_no_jump():
    cfg = Simcon()
    rng = np.random.default_rng(0)
    S, fam = build_sim(cfg, rng)
    z, labels, durs, onsets, bounds = _label_seq(cfg, 2, rng)
    M = _mean_logits(cfg, S, z, labels, durs, onsets)
    end_of_sign_1 = bounds[0] - 1
    start_of_sign_2 = bounds[0]
    assert np.allclose(M[end_of_sign_1], M[start_of_sign_2]), \
        "co-articulation blend should start exactly where the previous sign ended" 


def test_protocol_generation_separates_policy_scoring_and_manifest_views():
    cfg = Simcon(K=4, fami=2, dur_r=(5, 8), blend=2)
    stream = generate_stream(
        cfg,
        np.random.default_rng(17),
        role="fit",
        scenario="abrupt-change",
        seed=17,
        within_seed_index=3,
        n_signs=3,
    )

    assert stream.policy.stream_id == stream.scoring.stream_id
    assert stream.manifest.stream_id == stream.policy.stream_id
    assert stream.manifest.role == "fit"
    assert stream.manifest.scenario == "abrupt-change"
    assert stream.manifest.policy_visible_field_names == (
        "stream_id", "posteriors"
    )
    assert not stream.policy.posteriors.flags.writeable
    assert len(stream.policy) == len(stream.scoring.truth)
