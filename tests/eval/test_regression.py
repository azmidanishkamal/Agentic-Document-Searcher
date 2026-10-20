import json
from pathlib import Path
from typing import Any

import pytest

from src.eval.regression import (
    EXIT_REGRESSION,
    EXIT_SETUP,
    BaselineError,
    compare,
    load_baseline,
    main,
)
from src.eval.runner import EvalConfig


def _record(qid: str, passed: bool, error: str | None = None) -> dict[str, Any]:
    return {
        "id": qid,
        "answer_score": {"passed": passed, "reason": f"{qid} reason"},
        "error": error,
    }


def _results(*records: dict[str, Any], cost: float = 0.01) -> dict[str, Any]:
    return {
        "run": {},
        "summary": {"cost_usd": {"total": cost}},
        "questions": list(records),
    }


def _write_baseline(
    root: Path,
    records: list[dict[str, Any]],
    backend: str = "weaviate",
    mode: str = "hybrid_reranked",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    data = {
        "run": {"config": {"backend": backend, "retrieval_mode": mode}},
        "questions": records,
    }
    (root / f"{backend}_{mode}.json").write_text(json.dumps(data), encoding="utf-8")


class FakeRun:
    """Stands in for run_eval: returns queued results and records each config."""

    def __init__(self, *results: dict[str, Any]) -> None:
        self.results = list(results)
        self.configs: list[EvalConfig] = []

    def __call__(self, config: EvalConfig) -> dict[str, Any]:
        self.configs.append(config)
        return self.results.pop(0)


def _main(tmp_path: Path, run: FakeRun, questions: str = "q01,q02") -> int:
    return main(
        [
            "--backend",
            "weaviate",
            "--retrieval-mode",
            "hybrid_reranked",
            "--questions",
            questions,
            "--baseline-dir",
            str(tmp_path / "baselines"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        run=run,
    )


def test_compare_flags_only_baseline_passes_that_now_fail() -> None:
    baseline = {
        "q01": _record("q01", True),
        "q02": _record("q02", True),
        "q09": _record("q09", False),
        "q18": _record("q18", True),
    }
    results = _results(
        _record("q01", True),
        _record("q02", False),
        _record("q09", False),
        _record("q18", False, error="agent: TooManyRequestsError"),
    )

    comparison = compare(results, baseline)

    assert comparison.regressions == ["q02"]
    assert comparison.errors == ["q18"]
    assert comparison.not_gated == ["q09"]
    assert comparison.failed == ["q18", "q02"]


def test_missing_baseline_names_the_combination_and_expected_path(
    tmp_path: Path,
) -> None:
    _write_baseline(tmp_path, [_record("q01", True)], backend="weaviate")

    with pytest.raises(BaselineError) as exc:
        load_baseline("pinecone", "dense", ("q01",), root=tmp_path)

    message = str(exc.value)
    assert "backend='pinecone'" in message and "retrieval_mode='dense'" in message
    assert "pinecone_dense.json" in message
    assert "weaviate_hybrid_reranked" in message  # lists what does exist


def test_baseline_recorded_for_another_combination_is_rejected(tmp_path: Path) -> None:
    _write_baseline(tmp_path, [_record("q01", True)], mode="hybrid")
    (tmp_path / "weaviate_hybrid.json").rename(tmp_path / "weaviate_dense.json")

    with pytest.raises(BaselineError, match="recorded with"):
        load_baseline("weaviate", "dense", ("q01",), root=tmp_path)


def test_baseline_missing_a_requested_question_is_rejected(tmp_path: Path) -> None:
    _write_baseline(tmp_path, [_record("q01", True)])

    with pytest.raises(BaselineError, match="q07"):
        load_baseline("weaviate", "hybrid_reranked", ("q01", "q07"), root=tmp_path)


def test_main_fails_before_running_anything_when_baseline_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = FakeRun()

    assert _main(tmp_path, run) == EXIT_SETUP
    assert run.configs == []
    assert "No baseline" in capsys.readouterr().err


def test_main_passes_when_nothing_regresses(tmp_path: Path) -> None:
    _write_baseline(
        tmp_path / "baselines", [_record("q01", True), _record("q02", True)]
    )
    run = FakeRun(_results(_record("q01", True), _record("q02", True)))

    assert _main(tmp_path, run) == 0
    assert len(run.configs) == 1
    assert run.configs[0].backend == "weaviate"
    assert run.configs[0].question_ids == ("q01", "q02")
    assert (tmp_path / "out" / "results.json").exists()
    assert not (tmp_path / "out" / "recheck.json").exists()


def test_regression_that_passes_on_recheck_does_not_fail(tmp_path: Path) -> None:
    _write_baseline(
        tmp_path / "baselines", [_record("q01", True), _record("q02", True)]
    )
    run = FakeRun(
        _results(_record("q01", True), _record("q02", False)),
        _results(_record("q02", True)),
    )

    assert _main(tmp_path, run) == 0
    assert run.configs[1].question_ids == ("q02",)  # only the regression reruns
    assert (tmp_path / "out" / "recheck.json").exists()


def test_regression_that_fails_recheck_fails_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_baseline(
        tmp_path / "baselines", [_record("q01", True), _record("q02", True)]
    )
    run = FakeRun(
        _results(_record("q01", True), _record("q02", False)),
        _results(_record("q02", False)),
    )

    assert _main(tmp_path, run) == EXIT_REGRESSION
    out = capsys.readouterr()
    assert "| q02 | pass | FAIL | FAIL | **regression** |" in out.out
    assert "q02 reason" in out.out


def test_errors_are_rechecked_and_fail_if_they_persist(tmp_path: Path) -> None:
    _write_baseline(tmp_path / "baselines", [_record("q01", True)])
    run = FakeRun(
        _results(_record("q01", False, error="agent: 429")),
        _results(_record("q01", False, error="agent: 429")),
    )

    assert _main(tmp_path, run, questions="q01") == EXIT_REGRESSION


def test_question_failing_in_baseline_is_not_gated(tmp_path: Path) -> None:
    _write_baseline(tmp_path / "baselines", [_record("q09", False)])
    run = FakeRun(_results(_record("q09", False)))

    assert _main(tmp_path, run, questions="q09") == 0
    assert len(run.configs) == 1


def test_summary_is_appended_to_github_step_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_baseline(tmp_path / "baselines", [_record("q01", True)])
    step_summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step_summary))

    _main(tmp_path, FakeRun(_results(_record("q01", True))), questions="q01")

    assert "| q01 | pass | pass |  | ok |" in step_summary.read_text(encoding="utf-8")
