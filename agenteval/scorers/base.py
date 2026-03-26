"""Abstract base scorer — all scorers extend this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI

from agenteval.schema.models import ScoreResult, ScoreLevel


class BaseScorer(ABC):
    """Base class for all scorers."""

    name: str = "base"
    is_llm: bool = False  # Whether this scorer uses LLM calls

    def __init__(self, provider: str = "openai", api_key: str = "", model: str = ""):
        self.provider = provider.lower()
        if self.is_llm:
            if self.provider == "anthropic":
                self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
                self.model = model or "claude-sonnet-4-20250514"
            else:
                self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
                self.model = model or "gpt-4o"

    @abstractmethod
    def score(
        self,
        input_text: str,
        output_text: str,
        expected: str = "",
        **kwargs,
    ) -> ScoreResult:
        """Score a single agent response."""
        pass

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM for scoring."""
        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )
            return resp.choices[0].message.content

    @staticmethod
    def _score_to_level(score: float, pass_threshold: float = 0.7) -> ScoreLevel:
        """Convert a numeric score to a pass/partial/fail level."""
        if score >= pass_threshold:
            return ScoreLevel.PASS
        elif score >= 0.4:
            return ScoreLevel.PARTIAL
        return ScoreLevel.FAIL
