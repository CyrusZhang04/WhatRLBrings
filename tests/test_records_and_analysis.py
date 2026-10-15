import os

import pytest

from wrb import records as rec_io
from wrb.analysis import aggregate
from wrb.analysis.annotate import bold_spans
from wrb.analysis.replacements import Replacement, categorize, positional_density, table5
from wrb.analysis.response_stats import degradation_flags
from wrb.analysis.tables import table3


def _block(accs, lengths=None, rates=None):
    n = len(accs)
    return rec_io.make_block(
        texts=["\\boxed{480}" if a else "\\boxed{1}" for a in accs],
        accs=accs,
        lengths=lengths or [100] * n,
        replace_rates=rates,
    )


def test_make_block_rejects_ragged_arrays():
    with pytest.raises(ValueError):
        rec_io.make_block(texts=["a", "b"], accs=[True], lengths=[1, 2])


def test_make_block_rejects_a_short_replace_rate_array():
    with pytest.raises(ValueError):
        rec_io.make_block(
            texts=["a", "b"], accs=[True, False], lengths=[1, 2], replace_rates=[0.04]
        )


def test_condition_names_skips_metadata():
    rec = rec_io.make_record("aime24", 0, "Q", "480", {"base": _block([True])})
    assert rec_io.condition_names(rec) == ["base"]


def test_problem_id_is_one_based():
    rec = rec_io.make_record("aime24", 10, "Q", "480", {"base": _block([True])})
    assert rec_io.problem_id(rec) == "aime24#11"


def test_group_of_pools_the_two_year_pairs():
    assert rec_io.group_of("aime24") == rec_io.group_of("aime25") == "aime24+25"
    assert rec_io.group_of("aime22") == rec_io.group_of("aime23") == "aime22+23"
    assert rec_io.group_of("math500") == "math500"


def test_round_trip_through_a_shard(tmp_path):
    recs = [
        rec_io.make_record("aime24", i, "Q{}".format(i), "480", {"base": _block([True, False])})
        for i in range(3)
    ]
    path = os.path.join(str(tmp_path), "aime24", "aime24-n2-shard0of1.jsonl")
    assert rec_io.write_shard(path, recs) == 3
    back = list(rec_io.read_run(str(tmp_path)))
    assert [r["qi"] for r in back] == [0, 1, 2]


def test_aggregate_builds_per_problem_records(tmp_path):
    recs = [
        rec_io.make_record(
            "aime24",
            0,
            "Q",
            "480",
            {
                "base": _block([True, False, False, False]),
                "rl_replace": _block([True, True, False, False], rates=[0.04] * 4),
            },
        )
    ]
    rec_io.write_shard(os.path.join(str(tmp_path), "aime24-n4-shard0of1.jsonl"), recs)
    out = aggregate.build(str(tmp_path))
    assert len(out) == 1
    assert out[0].pass1("base") == 0.25
    assert out[0].pass1("rl_replace") == 0.5
    assert out[0].replace_rates["rl_replace"] == [0.04] * 4


def test_merge_refuses_a_condition_collision():
    a = [aggregate.ProblemRecord("aime24#1", "aime24", 0, "aime24+25", "Q", "480", {"base": [True]})]
    b = [aggregate.ProblemRecord("aime24#1", "aime24", 0, "aime24+25", "Q", "480", {"base": [False]})]
    with pytest.raises(ValueError):
        aggregate.merge(a, b)


def test_select_keeps_only_problems_with_every_condition():
    rs = [
        aggregate.ProblemRecord("p1", "aime24", 0, "aime24+25", "Q", "1", {"base": [True], "x": [True]}),
        aggregate.ProblemRecord("p2", "aime24", 1, "aime24+25", "Q", "1", {"base": [True]}),
    ]
    assert [r.problem for r in aggregate.select(rs, None, ["base", "x"])] == ["p1"]


def test_table3_recovers_the_expected_share():
    # Two problems; rl-replace sits exactly halfway between base and RL.
    rs = [
        aggregate.ProblemRecord(
            "p{}".format(i),
            "aime24",
            i,
            "aime24+25",
            "Q",
            "1",
            {
                "base": [True, False, False, False],
                "rl_replace": [True, True, False, False],
                "full_rl": [True, True, True, False],
            },
        )
        for i in range(2)
    ]
    rows = table3(rs, k=4, groups=("aime24+25",))
    assert rows[0]["pass1_base"] == "25.00"
    assert rows[0]["pass1_method"] == "50.00"
    assert rows[0]["pass1_rl"] == "75.00"
    assert rows[0]["recovery"].startswith("50.0")


def test_categorize_precedence():
    assert categorize("Correct", "Re") == "re_prefix"
    assert categorize("Let", "###") == "markdown_headers"
    assert categorize("So", "Wait") == "opening_and_pause_words"
    # The RL token decides: a pause word inserted where a header was is an
    # insertion of "Wait", not a markdown header (§3.5, "### -> Wait 30").
    assert categorize("###", "Wait") == "opening_and_pause_words"
    # ...but a header replaced by something unremarkable stays a header.
    assert categorize("###", "the") == "markdown_headers"
    assert categorize("we", "I") == "first_person_pronouns"
    assert categorize("12", "20") == "numbers"
    assert categorize("x", "\\frac") == "latex_formatting"
    assert categorize(".", ",") == "punctuation"
    assert categorize("compute", "calculate") == "other_content_words"


def test_table5_counts_only_text_changing_replacements():
    reps = [
        Replacement(0, "So", "Wait"),
        Replacement(1, "same", "same"),  # tokenization only
        Replacement(2, "compute", "calculate"),
    ]
    summary = table5(reps)
    assert summary["total"] == 2
    assert summary["n_tokenization_only"] == 1
    shares = {r["category"]: r["share_pct"] for r in summary["rows"]}
    assert shares["opening_and_pause_words"] == pytest.approx(50.0)
    assert shares["other_content_words"] == pytest.approx(50.0)


def test_positional_density_bins_by_relative_position():
    reps = [Replacement(i, "a", "b", rel_pos=0.05) for i in range(3)]
    reps += [Replacement(9, "a", "b", rel_pos=0.65)]
    dens = positional_density(reps, n_bins=10)
    assert dens[0]["count"] == 3
    assert dens[6]["count"] == 1
    assert sum(d["count"] for d in dens) == 4


def test_degradation_flags():
    at_limit = degradation_flags("\\boxed{1}", length=8192, max_tokens=8192)
    assert at_limit["flag_length"] and not at_limit["flag_no_answer"]

    no_answer = degradation_flags("I give up", length=500, max_tokens=8192)
    assert no_answer["flag_no_answer"]

    repeated = degradation_flags("ab" * 200, length=400, max_tokens=8192)
    assert repeated["flag_repetition"]


def test_bold_spans_keeps_whitespace_outside_the_markers():
    assert bold_spans("a Wait b", [(1, 6)]) == "a **Wait** b"


def test_bold_spans_without_spans_is_identity():
    assert bold_spans("unchanged", []) == "unchanged"
