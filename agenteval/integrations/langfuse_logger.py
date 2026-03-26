"""Langfuse integration — logs evaluation traces for observability."""
from __future__ import annotations

import logging
from typing import Optional

from agenteval.schema.models import EvalReport, EvalResult

logger = logging.getLogger(__name__)


class LangfuseLogger:
    """Logs evaluation results to Langfuse for observability."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        try:
            from langfuse import Langfuse
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host or "https://cloud.langfuse.com",
            )
            self.available = True
        except ImportError:
            logger.warning(
                "langfuse not installed. Install with: pip install agenteval-sdk[langfuse]"
            )
            self.available = False
            self.client = None

    def log_result(self, result: EvalResult) -> None:
        """Log a single evaluation result as a Langfuse trace."""
        if not self.available:
            return

        trace = self.client.trace(
            name=f"eval_{result.test_id}",
            metadata={
                "test_id": result.test_id,
                "pass_fail": result.pass_fail.value,
                "weighted_score": result.weighted_score,
                "latency_ms": result.latency_ms,
            },
        )

        trace.generation(
            name="agent_response",
            input=result.input,
            output=result.agent_output,
            metadata={"expected": result.expected},
        )

        for score in result.scores:
            trace.score(
                name=score.scorer_name,
                value=score.score,
                comment=score.reasoning,
            )

    def log_report(self, report: EvalReport) -> None:
        """Log an entire evaluation report."""
        if not self.available:
            return

        for result in report.results:
            self.log_result(result)

        self.client.flush()
        logger.info(f"Logged {len(report.results)} traces to Langfuse")
