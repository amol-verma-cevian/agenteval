"""Tests for schema models."""

from agenteval.schema.models import (
    TestInput, ScoreResult, EvalResult, EvalReport,
    ScoreLevel,
)


def test_test_input_defaults():
    t = TestInput(id="t1", input="hello")
    assert t.expected == ""
    assert t.category == "general"
    assert t.metadata == {}


def test_score_result():
    s = ScoreResult(
        scorer_name="relevance",
        score=0.85,
        level=ScoreLevel.PASS,
        reasoning="Good answer",
    )
    assert s.score == 0.85
    assert s.level == ScoreLevel.PASS


def test_eval_result():
    r = EvalResult(
        test_id="t1",
        input="hello",
        agent_output="Hi there!",
        weighted_score=0.75,
        pass_fail=ScoreLevel.PASS,
    )
    assert r.weighted_score == 0.75


def test_eval_report_pass_rate():
    report = EvalReport(
        total_cases=10,
        passed=7,
        partial=2,
        failed=1,
    )
    assert report.pass_rate == 70.0


def test_eval_report_empty():
    report = EvalReport()
    assert report.pass_rate == 0.0


def test_score_level_values():
    assert ScoreLevel.PASS.value == "pass"
    assert ScoreLevel.PARTIAL.value == "partial"
    assert ScoreLevel.FAIL.value == "fail"
