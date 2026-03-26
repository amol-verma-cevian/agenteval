"""Latency scorer — deterministic check for response time."""
from __future__ import annotations

from agenteval.scorers.base import BaseScorer
from agenteval.schema.models import ScoreResult


class LatencyScorer(BaseScorer):
    name = "latency"
    is_llm = False

    def score(self, input_text: str, output_text: str, expected: str = "", **kwargs) -> ScoreResult:
        latency_ms = kwargs.get("latency_ms", 0.0)
        threshold_ms = kwargs.get("latency_threshold_ms", 5000.0)

        if latency_ms <= 0:
            return ScoreResult(
                scorer_name=self.name,
                score=1.0,
                level=self._score_to_level(1.0),
                reasoning="No latency data provided",
            )

        if latency_ms <= threshold_ms * 0.5:
            score = 1.0
            reasoning = f"Excellent: {latency_ms:.0f}ms (well under {threshold_ms:.0f}ms threshold)"
        elif latency_ms <= threshold_ms:
            score = 0.7
            reasoning = f"Acceptable: {latency_ms:.0f}ms (under {threshold_ms:.0f}ms threshold)"
        elif latency_ms <= threshold_ms * 1.5:
            score = 0.4
            reasoning = f"Slow: {latency_ms:.0f}ms (exceeds {threshold_ms:.0f}ms threshold)"
        else:
            score = 0.1
            reasoning = f"Very slow: {latency_ms:.0f}ms (far exceeds {threshold_ms:.0f}ms threshold)"

        return ScoreResult(
            scorer_name=self.name,
            score=score,
            level=self._score_to_level(score),
            reasoning=reasoning,
        )
