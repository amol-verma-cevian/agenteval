"""Tests for CLI commands."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from agenteval.cli.main import cli


def _make_test_file():
    data = {
        "test_cases": [
            {"id": "t1", "input": "hello", "expected": "greeting", "category": "happy_path"},
        ]
    }
    path = tempfile.mktemp(suffix=".json")
    Path(path).write_text(json.dumps(data))
    return path


def _make_report_file():
    data = {
        "agent_name": "test",
        "evaluated_at": "2024-01-01",
        "total_cases": 1,
        "passed": 1,
        "partial": 0,
        "failed": 0,
        "avg_score": 0.85,
        "avg_latency_ms": 100.0,
        "results": [],
        "scorer_weights": {},
    }
    path = tempfile.mktemp(suffix=".json")
    Path(path).write_text(json.dumps(data))
    return path


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AgentEval" in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--tests" in result.output


def test_show_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "--help"])
    assert result.exit_code == 0
    assert "--report" in result.output


def test_scorers_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["scorers"])
    assert result.exit_code == 0
    assert "relevance" in result.output
    assert "format" in result.output


def test_show_command():
    report_path = _make_report_file()
    runner = CliRunner()
    result = runner.invoke(cli, ["show", "--report", report_path])
    assert result.exit_code == 0


def test_run_with_echo_agent():
    test_path = _make_test_file()
    output_path = tempfile.mktemp(suffix=".json")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "run",
        "--tests", test_path,
        "--output", output_path,
        "--scorers", "format",
        "--scorers", "latency",
    ])

    assert result.exit_code == 0
    data = json.loads(Path(output_path).read_text())
    assert data["total_cases"] == 1
