import math

import numpy as np
import pytest

from reliability_lab.evaluation import statistical_analysis as st


def test_wilson_known_value():
    lo, hi = st.wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert all(math.isnan(x) for x in st.wilson_ci(0, 0))


def test_wilson_extremes_in_unit_interval():
    lo, hi = st.wilson_ci(10, 10)
    assert 0 <= lo < hi <= 1


def test_mcnemar_exact():
    a = [True] * 8 + [False] * 2 + [True] * 5
    b = [False] * 8 + [False] * 2 + [True] * 5
    r = st.mcnemar_exact(a, b)
    assert (r["n10"], r["n01"]) == (8, 0)
    assert r["p_value"] == pytest.approx(2 * 0.5**8)
    assert r["diff"] > 0
    assert math.isinf(r["odds_ratio"])


def test_mcnemar_no_discordance_p_is_one():
    r = st.mcnemar_exact([True, False, True], [True, False, True])
    assert r["p_value"] == 1.0


def test_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        st.mcnemar_exact([True], [True, False])


def test_holm():
    adj = st.holm_correction([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
    assert all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))


def test_cohens_h():
    assert st.cohens_h(0.5, 0.5) == 0
    assert st.cohens_h(0.9, 0.5) > 0.8  # large effect


def test_rate_with_ci_one_row_per_case_uses_wilson():
    r = st.rate_with_ci([1, 1, 0, 1], clusters=["a", "b", "c", "d"])
    assert r["value"] == 0.75 and r["n"] == 4
    assert (r["ci_low"], r["ci_high"]) == pytest.approx(st.wilson_ci(3, 4))


def test_rate_with_ci_cluster_bootstrap_wider_than_naive():
    # 10 cases x 5 repeats, outcomes perfectly correlated within a case
    vals, cl = [], []
    for c in range(10):
        vals += [1.0 if c < 5 else 0.0] * 5
        cl += [c] * 5
    r = st.rate_with_ci(vals, cl, n_resamples=500, seed=1)
    naive = st.wilson_ci(25, 50)
    assert (r["ci_high"] - r["ci_low"]) > (naive[1] - naive[0])


def test_rate_with_ci_skips_nan():
    r = st.rate_with_ci([1, None, float("nan"), 0], clusters=[1, 2, 3, 4])
    assert r["n"] == 2 and r["value"] == 0.5


def test_ece_and_brier():
    assert st.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert st.expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)
    assert st.brier_score([1.0, 0.0], [True, False]) == 0.0
    assert st.brier_score([1.0], [False]) == 1.0


def test_kappa_perfect_and_chance():
    assert st.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert st.cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_paired_bootstrap_diff_sign():
    r = st.paired_bootstrap_diff([1, 1, 1, 0], [0, 0, 1, 0], clusters=[1, 2, 3, 4], n_resamples=300)
    assert r["diff"] == 0.5
