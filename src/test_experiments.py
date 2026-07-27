from evidence import Simcon
from experiments import one, summarize


def test_small_end_to_end_run():
    cfg = Simcon(
        K=4,
        fami=2,
        dur_r=(8, 12),
        blend=2,
    )

    n = {
        "sf": 10,
        "sd": 8,
        "st": 10,
        "cf": 6,
        "ct": 5,
    }

    a = one(11, cfg, n)
    b = one(12, cfg, n)

    assert set(a["cont"]) == {
        "M0", "M1", "M2", "M3", "MT"
    }

    for k in a["cont"]:
        assert a["cont"][k]["bnd_n"] == 20
        assert a["cont"][k]["hit_n"] >= 0
        assert a["cont"][k]["miss_n"] >= 0

    sm = summarize([a, b], nb=100)

    assert set(sm["cont"]) == {
        "M0", "M1", "M2", "M3", "MT"
    }

    assert sm["cont"]["M2"]["cov"]["n"] == 2
    assert sm["front"]["0.05"]["dp"]["risk"]["n"] == 2
    assert sm["front"]["0.05"]["immediate"]["latency"]["mean"] == 0.0
    paired = sm["front"]["0.05"]["paired_differences"]
    assert set(paired) == {"dp_minus_myopic", "dp_minus_fixed"}
    assert paired["dp_minus_myopic"]["risk"]["n"] == 2
    assert a["front"][1]["dp_label"] == (
        "empirical finite-horizon dynamic-programming approximation"
    )
    expected_ids = a["front"][1]["ordered_test_stream_ids"]
    assert len(expected_ids) == n["st"]
