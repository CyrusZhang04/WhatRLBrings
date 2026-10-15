import pytest

from wrb.grading import ends_in_repetition, has_final_answer, last_boxed, normalize_answer
from wrb.prompts import (
    ANSWER_INSTRUCTION,
    SYSTEM_MESSAGE,
    build_eval_prompt,
    chat_messages,
    qwen3_eval_prompt,
    rl_training_prompt,
    user_message,
)


def test_last_boxed_takes_the_last_one():
    assert last_boxed(r"first \boxed{12} then \boxed{480}") == "480"


def test_last_boxed_handles_nested_braces():
    assert last_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_last_boxed_returns_none_when_truncated_inside_the_box():
    # The common case at the generation limit.
    assert last_boxed(r"the answer is \boxed{48") is None
    assert last_boxed("no box here") is None


def test_has_final_answer_tracks_last_boxed():
    assert has_final_answer(r"\boxed{7}")
    assert not has_final_answer(r"\boxed{7")


def test_normalize_answer_strips_latex_decoration():
    assert normalize_answer(r"\left\frac{1}{2}\right") == r"\frac12"
    assert normalize_answer("$480$") == "480"
    assert normalize_answer(r"\text{480}") == "480"
    assert normalize_answer("480.") == "480"


def test_normalize_answer_passes_through_none_and_empties():
    assert normalize_answer(None) is None
    assert normalize_answer("   ") is None


def test_ends_in_repetition_catches_a_degenerate_tail():
    text = "a proper solution. " + "so the answer is " * 40
    assert ends_in_repetition(text)


def test_ends_in_repetition_leaves_normal_prose_alone():
    text = (
        "We place the vertices at (x, y), (-x, y), (-x, -y) and (x, -y), which "
        "describes a rectangle rather than a general rhombus, so the diagonals "
        "need not be perpendicular and the conclusion does not follow."
    )
    assert not ends_in_repetition(text)


def test_eval_prompt_is_the_appendix_d_string():
    prompt = qwen3_eval_prompt("What is 1+1?")
    assert prompt == (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is 1+1?\n"
        "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def test_user_message_joins_question_and_instruction():
    assert user_message("  Q?  ") == "Q?\n" + ANSWER_INSTRUCTION


def test_chat_messages_carry_the_two_turns():
    msgs = chat_messages("Q?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_MESSAGE


def test_prefill_is_appended_after_the_assistant_header():
    plain = build_eval_prompt("Q?")
    forced = build_eval_prompt("Q?", prefill="Alright")
    assert forced == plain + "Alright"


def test_non_qwen3_family_requires_a_tokenizer():
    with pytest.raises(ValueError):
        build_eval_prompt("Q?", family="llama3")


def test_rl_training_prompt_is_the_dapo_format():
    out = rl_training_prompt("Q?")
    assert out.startswith("Solve the following math problem step by step.")
    assert "Q?" in out
    assert out.rstrip().endswith('after "Answer:".')
    # The evaluation instruction must not leak into the training prompt.
    assert "\\boxed" not in out
