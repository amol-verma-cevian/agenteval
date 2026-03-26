"""Pydantic models — the data contract for evaluation."""
from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ScoreLevel(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


class TestInput(BaseModel):
    """A single test case input (from Synthia or manual)."""
    id: str
    input: str
    expected: str = ""
    category: str = "general"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """Result from a single scorer on a single test case."""
    scorer_name: str
    score: float  # 0.0 to 1.0
    level: ScoreLevel
    reasoning: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Complete evaluation result for a single test case."""
    test_id: str
    input: str
    agent_output: str
    expected: str = ""
    scores: list[ScoreResult] = Field(default_factory=list)
    weighted_score: float = 0.0
    pass_fail: ScoreLevel = ScoreLevel.FAIL
    latency_ms: float = 0.0


class EvalReport(BaseModel):
    """Complete evaluation report — what AgentEval outputs."""
    agent_name: str = "unknown"
    evaluated_at: str = ""
    total_cases: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    results: list[EvalResult] = Field(default_factory=list)
    scorer_weights: Dict[str, float] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return round(self.passed / self.total_cases * 100, 1)
