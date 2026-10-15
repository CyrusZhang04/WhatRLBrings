import math

import pytest

from wrb.analysis.aggregate import ProblemRecord
from wrb.analysis.mixture import (
    matching_fraction,
    mixture_curve,
    mixture_pass_at_k,
    mixture_pass_at_k_empirical,
)


def test_matching_fraction_reproduces_the_papers_92_percent():
    # Table 11's 512-sample values: base 10.26, RL 19.77, rl-replace 18.98.
    p = matching_fraction(0.1026, 0.1977, 0.1898)
    assert p == pytest.approx(0.917, abs=5e-3)


def test_matching_fraction_is_nan_when_the_models_tie():
    assert math.isnan(matching_fraction(0.2, 0.2, 0.25))


def test_matching_fraction_clamps_an_unreachable_target():
    # A target above the RL model cannot be reached by any mixture.
    assert matching_fraction(0.1, 0.2, 0.5) == 1.0
    assert matching_fraction(0.1, 0.2, 0.0) == 0.0
    assert matching_fraction(0.1, 0.2, 0.5, clamp=False) > 1.0


def test_mixture_pass_at_1_is_linear_in_the_fraction():
    base = [True, False, False, False]  # 0.25
    rl = [True, True, False, False]  # 0.50
    assert mixture_pass_at_k(base, rl, 0.0, 1) == pytest.approx(0.25)
    assert mixture_pass_at_k(base, rl, 1.0, 1) == pytest.approx(0.50)
    assert mixture_pass_at_k(base, rl, 0.5, 1) == pytest.approx(0.375)


def test_mixture_pass_at_k_is_monotone_in_k():
    base = [True, False, False, False]
    rl = [True, True, False, False]
    vals = [mixture_pass_at_k(base, rl, 0.9, k) for k in (1, 4, 16, 64)]
    assert vals == sorted(vals)
    assert vals[-1] <= 1.0


def test_mixture_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError):
        mixture_pass_at_k([True], [True], 1.5, 1)


def test_empirical_estimator_agrees_with_the_analytic_one():
    base = [True, False, False, False]
    rl = [True, True, False, False]
    for k in (1, 4, 8):
        a = mixture_pass_at_k(base, rl, 0.92, k)
        e = mixture_pass_at_k_empirical(base, rl, 0.92, k, n_trials=20000, seed=3)
        assert e == pytest.approx(a, abs=0.02)


def _records(n_problems=8):
    return [
        ProblemRecord(
            "p{}".format(i),
            "aime24",
            i,
            "aime24+25",
            "Q",
            "480",
            {
                "base": [True] + [False] * 15,
                "rl_replace": [True, True] + [False] * 14,
                "full_rl": [True, True, True] + [False] * 13,
            },
        )
        for i in range(n_problems)
    ]


def test_mixture_curve_matches_the_target_at_k_1():
    out = mixture_curve(_records(), ks=(1, 16))
    # The fraction is solved so that the mixture equals rl_replace at k=1.
    assert out["curve"]["k=1"] == pytest.approx(100 * 2 / 16, abs=1e-6)
    assert 0.0 <= out["p_rl"] <= 1.0
    assert out["matched_to"]["condition"] == "rl_replace"


def test_mixture_curve_reports_a_missing_condition_instead_of_raising():
    rs = [ProblemRecord("p0", "aime24", 0, "aime24+25", "Q", "1", {"base": [True]})]
    assert "error" in mixture_curve(rs)


def test_mixture_curve_accepts_an_explicit_fraction():
    out = mixture_curve(_records(), p_rl=1.0, match=None, ks=(1,))
    # All responses from the RL model: the mixture is the RL model.
    assert out["curve"]["k=1"] == pytest.approx(100 * 3 / 16, abs=1e-6)
    assert out["matched_to"] is None
