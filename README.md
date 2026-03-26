# AgentEval — Open-Source LLM Evaluation SDK

Evaluate your LLM chat agents with a combination of LLM-as-judge scorers and deterministic checks. AgentEval provides weighted scoring, CI/CD integration, and Langfuse observability out of the box.

## Why AgentEval?

LLM agents are non-deterministic. You can't just assert `output == expected`. AgentEval solves this with:

- **LLM-as-Judge scorers**: Relevance, tone, factual accuracy — scored by a judge LLM
- **Deterministic scorers**: Format checks, latency, refusal detection — no LLM needed
- **Weighted scoring engine**: Combine scores with configurable weights
- **Synthia integration**: Directly consume test cases from Synthia's output
- **Langfuse logging**: Ship evaluation traces for observability
- **CI/CD ready**: Exit code 1 when pass rate drops below threshold

## Architecture

```
test_cases.json (from Synthia)
       │
       ▼
┌──────────────┐
│   Evaluator   │
│   Engine      │
│               │
│  ┌──────────┐│     ┌──────────────┐
│  │ LLM      ││     │  Agent       │
│  │ Scorers  ││────▶│  (HTTP/fn)   │
│  ├──────────┤│     └──────────────┘
│  │ Determ.  ││
│  │ Scorers  ││
│  └──────────┘│
└──────────────┘
       │
       ▼
  eval_results.json ──▶ Langfuse / CI
```

## Quick Start

```bash
pip install agenteval-sdk

# Run evaluation (with echo agent for demo)
agenteval run \
  --tests test_cases.json \
  --output eval_results.json \
  --agent-name my-agent

# Run against your agent's HTTP endpoint
agenteval run \
  --tests test_cases.json \
  --agent-url http://localhost:8000/chat/message \
  --output eval_results.json

# View saved report
agenteval show --report eval_results.json

# List available scorers
agenteval scorers
```

## Scorers

| Scorer | Type | Weight | What It Measures |
|--------|------|--------|------------------|
| **Relevance** | LLM-as-Judge | 30% | Does the response address the query? |
| **Factual** | LLM-as-Judge | 25% | Are facts accurate? Any hallucinations? |
| **Tone** | LLM-as-Judge | 15% | Is the tone professional and appropriate? |
| **Format** | Deterministic | 15% | No errors, placeholders, or empty responses |
| **Latency** | Deterministic | 10% | Response time within acceptable threshold |
| **Refusal** | Deterministic | 5% | Correct refusal of adversarial inputs |

## Programmatic Usage

```python
from agenteval.engine.evaluator import Evaluator
from agenteval.schema.models import TestInput

# Define your agent
def my_agent(input_text: str) -> str:
    # Call your agent here
    return agent.respond(input_text)

# Load tests (from Synthia output)
tests = Evaluator.load_tests("test_cases.json")

# Run evaluation
evaluator = Evaluator(
    scorers=["relevance", "format", "latency"],
    provider="openai",
    pass_threshold=0.7,
)
report = evaluator.evaluate_suite(tests, my_agent, agent_name="my-agent")

print(f"Pass rate: {report.pass_rate}%")
print(f"Avg score: {report.avg_score}")

# Export
Evaluator.export_report(report, "eval_results.json")
```

## Langfuse Integration

```bash
pip install agenteval-sdk[langfuse]

# Set environment variables
export LANGFUSE_PUBLIC_KEY=pk-...
export LANGFUSE_SECRET_KEY=sk-...

# Run with Langfuse logging
agenteval run --tests test_cases.json --langfuse
```

## CI/CD Integration

AgentEval exits with code 1 when pass rate drops below 50%, making it easy to integrate into CI pipelines:

```yaml
# .github/workflows/eval.yml
- name: Run Agent Evaluation
  run: |
    agenteval run \
      --tests test_cases.json \
      --agent-url ${{ secrets.AGENT_URL }} \
      --scorers format latency refusal
```

## Pipeline

```
Synthia (generate) → test_cases.json → AgentEval (run) → eval_results.json → Dashboard/Langfuse
```

## Development

```bash
git clone https://github.com/amol-verma-cevian/agenteval.git
cd agenteval
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
agenteval/
├── agenteval/
│   ├── cli/main.py              # Click CLI (run, show, scorers)
│   ├── engine/evaluator.py      # Core evaluation engine
│   ├── scorers/                 # 6 scorers (3 LLM + 3 deterministic)
│   │   ├── base.py             # Abstract base with dual LLM support
│   │   ├── relevance.py        # LLM-as-judge relevance
│   │   ├── tone.py             # LLM-as-judge tone
│   │   ├── factual.py          # LLM-as-judge factual accuracy
│   │   ├── format_check.py     # Deterministic format checks
│   │   ├── latency.py          # Deterministic latency check
│   │   └── refusal.py          # Deterministic refusal detection
│   ├── integrations/
│   │   └── langfuse_logger.py  # Langfuse observability
│   └── schema/models.py        # Pydantic data contracts
├── tests/                       # 31 tests
└── pyproject.toml
```

## License

MIT
