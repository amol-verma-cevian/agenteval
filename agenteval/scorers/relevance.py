"""Relevance scorer — LLM-as-judge for response relevance."""
from __future__ import annotations

import json
import logging

from agenteval.scorers.base import BaseScorer
from agenteval.schema.models import ScoreResult

logger = logging.getLogger(__name__)


class RelevanceScorer(BaseScorer):
    name = "relevance"
    is_llm = True

    def score(self, input_text: str, output_text: str, expected: str = "", **kwargs) -> ScoreResult:
        prompt = f"""You are an evaluation judge. Score the relevance of an AI agent's response to the user's query.

User query: {input_text}
Agent response: {output_text}
Expected behavior: {expected or "Not specified"}

Score from 0.0 to 1.0:
- 1.0: Perfectly relevant, directly addresses the query
- 0.7: Mostly relevant, addresses the core query with minor gaps
- 0.4: Partially relevant, touches on the topic but misses key aspects
- 0.0: Completely irrelevant or off-topic

Return ONLY a JSON object: {{"score": 0.8, "reasoning": "brief explanation"}}"""

        try:
            raw = self._call_llm(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            s = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Relevance scorer failed: {e}")
            s = 0.0
            reasoning = f"Scoring failed: {e}"

        return ScoreResult(
            scorer_name=self.name,
            score=s,
            level=self._score_to_level(s),
            reasoning=reasoning,
        )
