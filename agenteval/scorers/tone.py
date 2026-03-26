"""Tone scorer — LLM-as-judge for response tone/professionalism."""
from __future__ import annotations

import json
import logging

from agenteval.scorers.base import BaseScorer
from agenteval.schema.models import ScoreResult

logger = logging.getLogger(__name__)


class ToneScorer(BaseScorer):
    name = "tone"
    is_llm = True

    def score(self, input_text: str, output_text: str, expected: str = "", **kwargs) -> ScoreResult:
        expected_tone = kwargs.get("expected_tone", "professional")

        prompt = f"""You are an evaluation judge. Score the tone of an AI agent's response.

User query: {input_text}
Agent response: {output_text}
Expected tone: {expected_tone}

Score from 0.0 to 1.0:
- 1.0: Perfect tone match — {expected_tone}, appropriate, and well-calibrated
- 0.7: Good tone — mostly appropriate with minor issues
- 0.4: Acceptable — tone is off but not harmful
- 0.0: Completely wrong tone — rude, overly casual, or inappropriate

Return ONLY a JSON object: {{"score": 0.8, "reasoning": "brief explanation"}}"""

        try:
            raw = self._call_llm(prompt)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            s = float(data.get("score", 0.0))
            reasoning = data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Tone scorer failed: {e}")
            s = 0.0
            reasoning = f"Scoring failed: {e}"

        return ScoreResult(
            scorer_name=self.name,
            score=s,
            level=self._score_to_level(s),
            reasoning=reasoning,
        )
