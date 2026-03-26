"""Format scorer — deterministic check for response structure."""
from __future__ import annotations

import re
from agenteval.scorers.base import BaseScorer
from agenteval.schema.models import ScoreResult


class FormatScorer(BaseScorer):
    name = "format"
    is_llm = False

    def score(self, input_text: str, output_text: str, expected: str = "", **kwargs) -> ScoreResult:
        checks = []
        issues = []

        # Check non-empty
        if output_text.strip():
            checks.append(1.0)
        else:
            checks.append(0.0)
            issues.append("Empty response")

        # Check reasonable length (not too short, not absurdly long)
        word_count = len(output_text.split())
        if 3 <= word_count <= 2000:
            checks.append(1.0)
        elif word_count < 3:
            checks.append(0.3)
            issues.append(f"Too short ({word_count} words)")
        else:
            checks.append(0.5)
            issues.append(f"Very long ({word_count} words)")

        # Check no raw error traces
        if "Traceback" in output_text or "Error:" in output_text:
            checks.append(0.0)
            issues.append("Contains error traces")
        else:
            checks.append(1.0)

        # Check no placeholder text
        placeholders = ["[INSERT", "TODO", "PLACEHOLDER", "lorem ipsum"]
        has_placeholder = any(p.lower() in output_text.lower() for p in placeholders)
        if has_placeholder:
            checks.append(0.0)
            issues.append("Contains placeholder text")
        else:
            checks.append(1.0)

        score = sum(checks) / len(checks) if checks else 0.0
        reasoning = "; ".join(issues) if issues else "All format checks passed"

        return ScoreResult(
            scorer_name=self.name,
            score=round(score, 2),
            level=self._score_to_level(score),
            reasoning=reasoning,
        )
