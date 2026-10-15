"""Prompt templates (paper Appendix D).

Every model and every intervention in the paper is evaluated with the same two
messages.  Keeping them in one place is deliberate: the recovery rates in
Table 1 are only comparable because nothing about the prompt changes between
conditions.

Two formats live here:

* the *evaluation* prompt, used for all of Tests 0-3 and the pass@k analysis;
* the *RL training* prompt, which is the format distributed with
  DAPO-Math-17k and is only used by the GRPO rollouts (``configs/rl_training``).

They are different, and mixing them up silently changes the numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYSTEM_MESSAGE = "You are a helpful assistant."

#: Appended to the problem statement, separated by a single newline.
ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

#: Prompt format distributed with DAPO-Math-17k, used for the GRPO rollouts
#: (paper Appendix D, "RL training").  Not used at evaluation time.
RL_TRAINING_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
    "{question}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

#: Literal evaluation prompt for Qwen3 models: the chat template applied with
#: thinking disabled, which inserts an empty thinking block.  Reproduced here so
#: the exact string is in the repository and does not depend on the tokenizer
#: version shipped with a particular checkpoint.  ``assert_matches_tokenizer``
#: checks the two against each other.
QWEN3_EVAL_PROMPT = (
    "<|im_start|>system\n"
    "{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "{question}\n"
    "{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


def user_message(question: str) -> str:
    """The user turn: the problem, then the answer-format instruction."""
    return "{}\n{}".format(question.strip(), ANSWER_INSTRUCTION)


def chat_messages(question: str) -> List[Dict[str, str]]:
    """The two messages every condition in the paper receives."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message(question)},
    ]


def qwen3_eval_prompt(question: str) -> str:
    """Evaluation prompt for Qwen3 models, built from the literal template."""
    return QWEN3_EVAL_PROMPT.format(
        system=SYSTEM_MESSAGE,
        question=question.strip(),
        instruction=ANSWER_INSTRUCTION,
    )


def build_eval_prompt(
    question: str,
    tokenizer: Any = None,
    prefill: Optional[str] = None,
    family: str = "qwen3",
) -> str:
    """Build the evaluation prompt, optionally prefilling the assistant turn.

    ``prefill`` is how Test 2 (§3.4) is run: the forced text ("Alright", or
    "Alright, I") is appended directly after the assistant header, and the base
    model continues from there with standard sampling.

    With ``tokenizer`` given, the prompt comes from that tokenizer's chat
    template, which is what the paper's runs used; Qwen3 models get
    ``enable_thinking=False``.  Without one, Qwen3 falls back to the literal
    template above and other families raise, because their templates are not
    reproduced here.
    """
    if tokenizer is None:
        if family != "qwen3":
            raise ValueError(
                "a tokenizer is required for family={!r}; only the Qwen3 "
                "evaluation prompt is reproduced literally in this module "
                "(paper Appendix D)".format(family)
            )
        prompt = qwen3_eval_prompt(question)
    else:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if family == "qwen3":
            # Qwen3's template inserts an empty thinking block when disabled.
            kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(chat_messages(question), **kwargs)

    if prefill:
        prompt = prompt + prefill
    return prompt


def rl_training_prompt(question: str) -> str:
    """Rollout prompt for GRPO training (DAPO-Math-17k format)."""
    return RL_TRAINING_TEMPLATE.format(question=question.strip())


def assert_matches_tokenizer(tokenizer: Any, question: str = "1+1=?") -> None:
    """Fail loudly if a checkpoint's chat template has drifted from Appendix D.

    Worth calling once per evaluation run: a tokenizer update that changes the
    thinking block or the header would otherwise shift every number quietly.
    """
    from_template = build_eval_prompt(question, tokenizer=tokenizer)
    literal = qwen3_eval_prompt(question)
    if from_template != literal:
        raise AssertionError(
            "chat template does not match the Appendix D prompt\n"
            "--- tokenizer ---\n{!r}\n--- expected ---\n{!r}".format(
                from_template, literal
            )
        )
