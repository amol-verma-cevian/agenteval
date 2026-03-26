"""Evaluation engine — orchestrates scorers and computes weighted results."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any

from agenteval.schema.models import (
    TestInput, EvalResult, EvalReport, ScoreResult, ScoreLevel,
)
from agenteval.scorers.base import BaseScorer
from agenteval.scorers import ALL_SCORERS

logger = logging.getLogger(__name__)

# Default weights for each scorer
DEFAULT_WEIGHTS = {
    "relevance": 0.30,
    "tone": 0.15,
    "factual": 0.25,
    "format": 0.15,
    "latency": 0.10,
    "refusal": 0.05,
}


class Evaluator:
    """Main evaluation engine — runs scorers against agent outputs."""

    def __init__(
        self,
        scorers: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        provider: str = "openai",
        api_key: str = "",
        model: str = "",
        pass_threshold: float = 0.7,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.pass_threshold = pass_threshold

        # Initialize scorers
        scorer_names = scorers or list(ALL_SCORERS.keys())
        self.scorers: Dict[str, BaseScorer] = {}
        for name in scorer_names:
            if name in ALL_SCORERS:
                self.scorers[name] = ALL_SCORERS[name](
                    provider=provider, api_key=api_key, model=model
                )

        # Set weights
        self.weights = weights or {k: DEFAULT_WEIGHTS.get(k, 0.1) for k in self.scorers}
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def evaluate_single(
        self,
        test: TestInput,
        agent_fn: Callable[[str], str],
        **kwargs,
    ) -> EvalResult:
        """Evaluate a single test case against the agent.

        Args:
            test: The test input to evaluate.
            agent_fn: A callable that takes user input and returns agent output.
            **kwargs: Additional context passed to scorers.

        Returns:
            EvalResult with scores from all scorers.
        """
        # Call agent and measure latency
        start = time.time()
        agent_output = agent_fn(test.input)
        latency_ms = (time.time() - start) * 1000

        # Run all scorers
        scores = []
        for name, scorer in self.scorers.items():
            scorer_kwargs = {**kwargs, **test.metadata}
            scorer_kwargs["latency_ms"] = latency_ms

            # Auto-detect adversarial for refusal scorer
            if name == "refusal" and test.category == "adversarial":
                scorer_kwargs["should_refuse"] = True

            result = scorer.score(
                input_text=test.input,
                output_text=agent_output,
                expected=test.expected,
                **scorer_kwargs,
            )
            scores.append(result)

        # Compute weighted score
        weighted = 0.0
        for s in scores:
            w = self.weights.get(s.scorer_name, 0.0)
            weighted += s.score * w

        # Determine pass/fail
        if weighted >= self.pass_threshold:
            level = ScoreLevel.PASS
        elif weighted >= 0.4:
            level = ScoreLevel.PARTIAL
        else:
            level = ScoreLevel.FAIL

        return EvalResult(
            test_id=test.id,
            input=test.input,
            agent_output=agent_output,
            expected=test.expected,
            scores=scores,
            weighted_score=round(weighted, 3),
            pass_fail=level,
            latency_ms=round(latency_ms, 1),
        )

    def evaluate_suite(
        self,
        tests: List[TestInput],
        agent_fn: Callable[[str], str],
        agent_name: str = "my-agent",
        **kwargs,
    ) -> EvalReport:
        """Evaluate an entire test suite.

        Args:
            tests: List of test inputs.
            agent_fn: Agent callable.
            agent_name: Name for the report.

        Returns:
            EvalReport with all results and aggregate stats.
        """
        results = []
        for test in tests:
            result = self.evaluate_single(test, agent_fn, **kwargs)
            results.append(result)
            logger.info(
                f"[{test.id}] score={result.weighted_score:.2f} "
                f"status={result.pass_fail.value}"
            )

        # Aggregate
        passed = sum(1 for r in results if r.pass_fail == ScoreLevel.PASS)
        partial = sum(1 for r in results if r.pass_fail == ScoreLevel.PARTIAL)
        failed = sum(1 for r in results if r.pass_fail == ScoreLevel.FAIL)
        avg_score = sum(r.weighted_score for r in results) / len(results) if results else 0.0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0.0

        return EvalReport(
            agent_name=agent_name,
            evaluated_at=datetime.now().isoformat(),
            total_cases=len(results),
            passed=passed,
            partial=partial,
            failed=failed,
            avg_score=round(avg_score, 3),
            avg_latency_ms=round(avg_latency, 1),
            results=results,
            scorer_weights=self.weights,
        )

    @staticmethod
    def load_tests(path: str) -> List[TestInput]:
        """Load test inputs from a JSON file (Synthia output or AgentEval format)."""
        data = json.loads(Path(path).read_text())

        # Support both formats
        if "test_cases" in data:
            # AgentEval format
            items = data["test_cases"]
        elif "cases" in data:
            # Synthia native format — flatten turns
            items = []
            for case in data["cases"]:
                for turn in case.get("turns", []):
                    items.append({
                        "id": f"{case['test_id']}_t{turn['turn']}",
                        "input": turn["input"],
                        "expected": turn.get("expected_behavior", ""),
                        "category": case.get("category", "general"),
                        "metadata": {
                            "test_id": case["test_id"],
                            "persona": case.get("persona", "generic"),
                            "difficulty": case.get("difficulty", "medium"),
                            "tags": case.get("tags", []),
                        },
                    })
        else:
            items = data if isinstance(data, list) else []

        return [TestInput(**item) for item in items]

    @staticmethod
    def export_report(report: EvalReport, path: str) -> str:
        """Export evaluation report to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report.model_dump(), indent=2, default=str))
        return str(p)
