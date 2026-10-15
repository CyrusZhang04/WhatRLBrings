import math

import pytest

from wrb.metrics import (
    bootstrap_ci,
    distinct_answers,
    distinct_openings,
    maj_at_k,
    pass_at_1,
    pass_at_k,
    pass_at_k_mean,
    paired_diff_ci,
    recovery_ci,
    recovery_rate,
)


def test_pass_at_1_is_the_mean():
    assert pass_at_1([True, False, False, False]) == 0.25
    assert math.isnan(pass_at_1([]))


@pytest.mark.parametrize(
    "n_correct,n,k,expected",
    [
        (0, 16, 16, 0.0),  # never solved
        (1, 16, 16, 1.0),  # k == n and one success
        (1, 4, 2, 0.5),  # 1 - C(3,2)/C(4,2)
        (2, 5, 2, 0.7),  # 1 - C(3,2)/C(5,2)
        (4, 4, 1, 1.0),  # always solved
    ],
)
def test_pass_at_k_matches_the_closed_form(n_correct, n, k, expected):
    correct = [True] * n_correct + [False] * (n - n_correct)
    assert pass_at_k(correct, k) == pytest.approx(expected)


def test_pass_at_k_is_monotone_in_k():
    correct = [True, False, False, False, False, False, False, False]
    vals = [pass_at_k(correct, k) for k in (1, 2, 4, 8)]
    assert vals == sorted(vals)


def test_pass_at_k_refuses_k_above_n():
    with pytest.raises(ValueError):
        pass_at_k([True, False], 4)


def test_pass_at_k_mean_averages_over_problems():
    per_problem = [[True, False], [False, False]]
    assert pass_at_k_mean(per_problem, 1) == pytest.approx(0.25)


def test_recovery_rate_equation_1():
    # The paper's headline: base 10.00, rl-replace 18.44, RL 19.32 -> 90.5%
    r = recovery_rate(0.1844, 0.1000, 0.1932)
    assert r == pytest.approx(0.905, abs=1e-3)


def test_recovery_rate_is_nan_without_an_rl_gain():
    assert math.isnan(recovery_rate(0.5, 0.3, 0.3))


def test_recovery_can_exceed_one():
    # MATH-500: rl-replace overshoots the RL model (106.6%)
    assert recovery_rate(0.8525, 0.7110, 0.8437) > 1.0


def test_maj_at_k_picks_the_most_frequent_answer():
    assert maj_at_k(["480", "480", "12"], "480") == 1.0
    assert maj_at_k(["12", "12", "480"], "480") == 0.0


def test_maj_at_k_ignores_responses_without_an_answer():
    assert maj_at_k([None, None, "480"], "480") == 1.0
    assert maj_at_k([None, None], "480") == 0.0


def test_maj_at_k_groups_equivalent_answers():
    def equivalent(a, b):
        return a.strip("0.") == b.strip("0.")

    # "480.0" and "480" are one group of two, beating "12"
    assert maj_at_k(["480.0", "480", "12"], "480", equivalent=equivalent) == 1.0


def test_maj_at_k_respects_k():
    answers = ["12", "12", "480", "480", "480"]
    assert maj_at_k(answers, "480", k=2) == 0.0
    assert maj_at_k(answers, "480") == 1.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.25, 0.5, 0.75, 1.0] * 8
    mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=7)
    assert lo <= mean <= hi
    assert bootstrap_ci(values, n_boot=500, seed=7) == (mean, lo, hi)


def test_bootstrap_ci_keeps_group_sizes():
    # One group is all zeros, the other all ones; a within-group resample can
    # never move the mean off 0.5.
    values = [0.0] * 30 + [1.0] * 30
    groups = ["a"] * 30 + ["b"] * 30
    mean, lo, hi = bootstrap_ci(values, groups=groups, n_boot=400, seed=1)
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_paired_diff_ci_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_diff_ci([0.1, 0.2], [0.1])


def test_recovery_ci_reports_degenerate_denominators():
    # RL never beats base here, so the ratio is meaningless on every resample.
    method = [0.5] * 20
    base = [0.4] * 20
    rl = [0.4] * 20
    out = recovery_ci(method, base, rl, n_boot=200)
    assert out["frac_denom_nonpositive"] == 1.0
    assert math.isnan(out["lo"]) and math.isnan(out["hi"])


def test_recovery_ci_brackets_the_point_estimate():
    base = [0.0, 0.1, 0.0, 0.2, 0.1] * 8
    method = [0.2, 0.3, 0.1, 0.4, 0.2] * 8
    rl = [0.25, 0.3, 0.1, 0.45, 0.2] * 8
    out = recovery_ci(method, base, rl, n_boot=1000, seed=3)
    assert out["lo"] <= out["recovery"] <= out["hi"]
    assert out["frac_denom_nonpositive"] == 0.0


def test_diversity_counts():
    texts = ["Alright, I have this", "Alright, I have this", "We are given"]
    assert distinct_openings(texts, n_chars=300) == 2
    assert distinct_answers(["480", "480", None, "12"]) == 2
