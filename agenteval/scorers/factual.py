"""Factual consistency scorer — LLM-as-judge for factual accuracy."""
from __future__ import annotations

import json
import logging

from agenteval.scorers.base import BaseScorer
from agenteval.schema.models import ScoreResult

logger = logging.getLogger(__name__)


class FactualScorer(BaseScorer):
    name = "factual"
    is_llm = True

    def score(self, input_text: str, output_text: str, expected: str = "", **kwargs) -> ScoreResult:
        context = kwargs.get("context", "")

        prompt = f"""You are an evaluation judge. Score the factual accuracy of an AI agent's response.

User query: {input_text}
Agent response: {output_text}
Expected answer: {expected or "Not specified"}
Context/ground truth: {context or "Not provided"}

Score from 0.0 to 1.0:
- 1.0: All claims are factually correct and verifiable
- 0.7: Mostly correct with minor inaccuracies
- 0.4: Mix of correct and incorrect information
- 0.0: Contains fabricated data or hallucinations

Focus on: Did the agent make up numbers? Did it claim things not in the data? Did it contradict known facts?

Return ONLY a JSON object: {{"score": 0.8, "reasoning": "brief explanation"}}"""

        try:
            raw = self._call_llm(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            s = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Factual scorer failed: {e}")
            s = 0.0
            reasoning = f"Scoring failed: {e}"

        return ScoreResult(
            scorer_name=self.name,
            score=s,
            level=self._score_to_level(s),
            reasoning=reasoning,
        )
