"""AgentEval scorers — LLM-as-judge and deterministic."""

from agenteval.scorers.base import BaseScorer
from agenteval.scorers.relevance import RelevanceScorer
from agenteval.scorers.tone import ToneScorer
from agenteval.scorers.factual import FactualScorer
from agenteval.scorers.format_check import FormatScorer
from agenteval.scorers.latency import LatencyScorer
from agenteval.scorers.refusal import RefusalScorer

ALL_SCORERS = {
    "relevance": RelevanceScorer,
    "tone": ToneScorer,
    "factual": FactualScorer,
    "format": FormatScorer,
    "latency": LatencyScorer,
    "refusal": RefusalScorer,
}

__all__ = [
    "BaseScorer",
    "RelevanceScorer",
    "ToneScorer",
    "FactualScorer",
    "FormatScorer",
    "LatencyScorer",
    "RefusalScorer",
    "ALL_SCORERS",
]
