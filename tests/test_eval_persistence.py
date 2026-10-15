"""Regression tests for saving completed questions before an evaluation fails."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("run_eval_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _: object())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(module, "load_model", lambda *args: object())
    monkeypatch.setattr(module, "build_eval_prompt", lambda question, **kwargs: question)
    monkeypatch.setattr(
        module, "Grader",
        lambda: types.SimpleNamespace(grade_many=lambda texts, answer: [True]),
    )

    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        "".join(json.dumps({"question": "Q{}".format(i), "answer": str(i)}) + "\n"
                for i in range(3)),
        encoding="utf-8",
    )
    out = tmp_path / "run"
    path = out / "aime24" / "aime24-n1-shard0of1.jsonl"
    argv = ["--condition", "base", "--base-model", "unused", "--problems", str(problems),
            "--split", "aime24", "--n-samples", "1", "--out", str(out)]
    return module, argv, path


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("failure", [None, "generation", "grading", "interrupt"])
def test_completed_questions_are_synced_before_next_question(evaluation, monkeypatch, failure):
    module, argv, path = evaluation
    synced_counts = []
    real_fsync = os.fsync

    def observe_fsync(fd):
        real_fsync(fd)
        synced_counts.append(len(read_records(path)))

    monkeypatch.setattr(module.rec_io.os, "fsync", observe_fsync)

    def generate(args, prompt, *models):
        index = int(prompt[1:])
        assert [r["qi"] for r in read_records(path)] == list(range(index))
        assert synced_counts == list(range(1, index + 1))
        if index == 1 and failure == "generation":
            raise RuntimeError("simulated CUDA out of memory")
        if index == 1 and failure == "interrupt":
            raise KeyboardInterrupt()
        return ["答案"], [1], None, None

    def grade(texts, answer):
        if answer == "1" and failure == "grading":
            raise RuntimeError("simulated grading failure")
        return [True]

    monkeypatch.setattr(module, "generate", generate)
    monkeypatch.setattr(module, "Grader", lambda: types.SimpleNamespace(grade_many=grade))
    if failure is None:
        module.main(argv)
        assert synced_counts == [1, 2, 3]
    else:
        error = KeyboardInterrupt if failure == "interrupt" else RuntimeError
        with pytest.raises(error):
            module.main(argv)
        assert synced_counts == [1]
    saved = read_records(path)
    assert [r["qi"] for r in saved] == ([0, 1, 2] if failure is None else [0])
    assert all(r["base"]["texts"] == ["答案"] for r in saved)


def test_rerun_preserves_existing_results_before_loading_models(evaluation, monkeypatch):
    module, argv, path = evaluation
    path.parent.mkdir(parents=True)
    original = b'{"previous": "completed result"}\n'
    path.write_bytes(original)

    def must_not_start(*args):
        pytest.fail("existing output must be rejected before loading models")

    monkeypatch.setattr(module, "load_model", must_not_start)
    with pytest.raises(FileExistsError):
        module.main(argv)
    assert path.read_bytes() == original


def test_sync_error_stops_before_next_question(evaluation, monkeypatch, capsys):
    module, argv, path = evaluation
    generated = []

    def generate(args, prompt, *models):
        generated.append(prompt)
        return ["answer"], [1], None, None

    def fail_sync(fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(module, "generate", generate)
    monkeypatch.setattr(module.rec_io.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="simulated disk failure"):
        module.main(argv)
    assert generated == ["Q0"]
    assert "(saved)" not in capsys.readouterr().out


def test_abrupt_exit_preserves_completed_question(evaluation):
    module, argv, path = evaluation
    # os._exit bypasses finally blocks, context-manager cleanup and buffer flushing.
    code = textwrap.dedent("""
        import os
        import sys
        import types
        from scripts import run_eval

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _: object())
        sys.modules["transformers"] = transformers
        run_eval.load_model = lambda *args: object()
        run_eval.build_eval_prompt = lambda question, **kwargs: question
        run_eval.Grader = lambda: types.SimpleNamespace(grade_many=lambda *args: [True])

        def generate(args, prompt, *models):
            if prompt == "Q1":
                os._exit(27)
            return ["completed"], [1], None, None

        run_eval.generate = generate
        run_eval.main(sys.argv[1:])
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, *argv],
        cwd=str(Path(module.__file__).resolve().parents[1]),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 27, result.stderr.decode("utf-8", errors="replace")
    saved = read_records(path)
    assert [r["qi"] for r in saved] == [0]
    assert saved[0]["base"]["texts"] == ["completed"]
