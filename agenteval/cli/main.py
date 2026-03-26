"""AgentEval CLI — the main entry point."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from agenteval.engine.evaluator import Evaluator
from agenteval.schema.models import EvalReport, ScoreLevel
from agenteval.scorers import ALL_SCORERS

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """AgentEval — Open-source LLM evaluation SDK."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(name)s | %(message)s")


@cli.command()
@click.option("--tests", "-t", required=True, help="Path to test cases JSON (from Synthia)")
@click.option("--agent-url", "-a", help="Agent HTTP endpoint URL")
@click.option("--output", "-o", default="eval_results.json", help="Output file path")
@click.option("--agent-name", default="my-agent", help="Agent name for the report")
@click.option("--scorers", "-s", multiple=True, help="Scorers to use (default: all)")
@click.option("--provider", default="openai", type=click.Choice(["openai", "anthropic"]))
@click.option("--model", default="", help="Model override for LLM scorers")
@click.option("--api-key", default="", help="API key")
@click.option("--pass-threshold", default=0.7, help="Score threshold for pass (0-1)")
@click.option("--langfuse", is_flag=True, help="Log results to Langfuse")
def run(tests, agent_url, output, agent_name, scorers, provider, model, api_key, pass_threshold, langfuse):
    """Run evaluation on test cases against an agent."""
    # Load tests
    test_path = Path(tests)
    if not test_path.exists():
        console.print(f"[red]Error: Test file not found: {tests}[/red]")
        sys.exit(1)

    test_inputs = Evaluator.load_tests(tests)

    # Create agent function
    if agent_url:
        agent_fn = _make_http_agent(agent_url)
    else:
        agent_fn = _make_echo_agent()
        console.print("[yellow]No --agent-url provided, using echo agent for demo[/yellow]")

    scorer_list = list(scorers) if scorers else None

    evaluator = Evaluator(
        scorers=scorer_list,
        provider=provider,
        api_key=api_key,
        model=model,
        pass_threshold=pass_threshold,
    )

    console.print(Panel(
        f"[bold]AgentEval[/bold]\n"
        f"Agent: {agent_name}\n"
        f"Tests: {len(test_inputs)} | Scorers: {len(evaluator.scorers)}\n"
        f"Provider: {provider} | Threshold: {pass_threshold}",
        title="Configuration",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Evaluating {len(test_inputs)} cases...", total=None)
        report = evaluator.evaluate_suite(test_inputs, agent_fn, agent_name=agent_name)
        progress.update(task, description=f"[green]Evaluation complete")

    # Export
    path = Evaluator.export_report(report, output)
    console.print(f"\n[green]Report saved → {path}[/green]")

    # Langfuse
    if langfuse:
        from agenteval.integrations.langfuse_logger import LangfuseLogger
        lf = LangfuseLogger()
        lf.log_report(report)

    # Print summary
    _print_report(report)

    # Exit with non-zero if too many failures
    if report.pass_rate < 50:
        sys.exit(1)


@cli.command()
@click.option("--report", "-r", required=True, help="Path to eval results JSON")
def show(report):
    """Display a saved evaluation report."""
    path = Path(report)
    if not path.exists():
        console.print(f"[red]Error: Report not found: {report}[/red]")
        sys.exit(1)

    data = json.loads(path.read_text())
    eval_report = EvalReport(**data)
    _print_report(eval_report)


@cli.command()
def scorers():
    """List available scorers."""
    table = Table(title="Available Scorers")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Description")

    descriptions = {
        "relevance": "LLM-as-judge for response relevance to query",
        "tone": "LLM-as-judge for response tone and professionalism",
        "factual": "LLM-as-judge for factual accuracy and hallucination detection",
        "format": "Deterministic checks for response structure and formatting",
        "latency": "Deterministic check for response time against threshold",
        "refusal": "Deterministic check for correct adversarial input refusal",
    }

    for name, cls in ALL_SCORERS.items():
        scorer_type = "LLM" if cls.is_llm else "Deterministic"
        desc = descriptions.get(name, "")
        table.add_row(name, scorer_type, desc)

    console.print(table)


# --- Helpers ---

def _make_http_agent(url: str):
    """Create an agent function that calls an HTTP endpoint."""
    import urllib.request
    import urllib.error

    def agent_fn(input_text: str) -> str:
        try:
            data = json.dumps({"input": input_text}).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return result.get("output", result.get("response", str(result)))
        except Exception as e:
            return f"Error calling agent: {e}"

    return agent_fn


def _make_echo_agent():
    """Create a simple echo agent for demo purposes."""
    def agent_fn(input_text: str) -> str:
        return f"I received your query: '{input_text}'. Let me help you with that."
    return agent_fn


def _print_report(report: EvalReport):
    """Print evaluation report as rich tables."""
    # Summary
    console.print(Panel(
        f"[bold]Pass Rate: {report.pass_rate}%[/bold]\n"
        f"Avg Score: {report.avg_score:.3f} | Avg Latency: {report.avg_latency_ms:.0f}ms\n"
        f"Passed: {report.passed} | Partial: {report.partial} | Failed: {report.failed}",
        title=f"Evaluation Report — {report.agent_name}",
    ))

    # Per-case results
    table = Table(title="Results")
    table.add_column("Test ID", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    table.add_column("Input", max_width=40)

    status_colors = {"pass": "green", "partial": "yellow", "fail": "red"}

    for r in report.results[:20]:  # Show first 20
        color = status_colors.get(r.pass_fail.value, "white")
        table.add_row(
            r.test_id,
            f"{r.weighted_score:.2f}",
            f"[{color}]{r.pass_fail.value}[/{color}]",
            r.input[:40] + "..." if len(r.input) > 40 else r.input,
        )

    if len(report.results) > 20:
        table.add_row("...", "", "", f"({len(report.results) - 20} more)")

    console.print(table)


if __name__ == "__main__":
    cli()
