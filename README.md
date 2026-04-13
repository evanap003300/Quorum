# Quorum

An AI physics and mathematics problem solver that decomposes problems into atomic, verifiable steps and executes them in a sandboxed environment. Built to fight LLM hallucination on graduate-level quantitative problems.

## The idea

LLMs are unreliable when they freehand multi-step math. Quorum wraps the LLM in a structured pipeline that forces it to work the way a careful student would: classify the problem, plan explicit steps, audit the plan, then execute each step as sandboxed Python with parallel agents voting on the result.

```
Problem ─► Router ─► {EASY:  single-agent (Flash)
                     {MEDIUM: single-agent (Pro)
                     {HARD:   Planner → Physics Lawyer → Revisor
                              → K-Ahead Swarm (3 parallel solvers, majority vote)
                              → E2B Python sandbox
                     ─► Final answer + full state trace
```

Key ideas:
- **Router** — `gemini-3-flash-preview` classifies problems into EASY/MEDIUM/HARD so we only spend the full orchestrator on problems that need it.
- **Atomic planning** — each step does exactly one thing (extract / calculate / convert).
- **Physics Lawyer + Revisor** — audits plans for reference-frame, conservation-law, and unit errors, then repairs them.
- **K-Ahead Swarm** — 3 parallel solver agents per step with majority voting on numeric results (1% tolerance). Resilient to single-agent syntax crashes and hallucinations.
- **SafeMath sandbox** — clamps `acos`, `asin`, `sqrt`, `log` inputs to avoid domain errors.
- **Vision module** — for image problems, GPT-4o iteratively applies CV tools (grid overlay, crop, binarize) to extract the problem before solving.

## Results

Evaluated on **SciBench** (college-level physics/math problems — the benchmark LLMs historically struggle with).

| Sprint | Accuracy           | Notes                                 |
|--------|--------------------|---------------------------------------|
| 6      | **78.9% (15/19)**  | Best result                           |
| 7      | 70% (14/20)        | Verification layer added then removed |
| 8      | in progress        | Math domain fixes, extended timeouts  |

### How baselines from the SciBench paper do

Numbers from the [SciBench paper (ICML 2024)](https://arxiv.org/html/2307.10635v2):

| System                              | Accuracy |
|-------------------------------------|----------|
| LLaMA-2-70B, zero-shot              | 2.4%     |
| LLaMA-2-70B, few-shot               | 8.4%     |
| GPT-4, zero-shot CoT                | ~30.4%   |
| GPT-4 + external tools (Python/Wolfram) | ~43.2% |
| Best prompting strategy in the paper | 48.96%  |
| **Quorum (Sprint 6)**               | **78.9%** |

Caveat: Quorum's 78.9% is measured on a 19-problem sample, while the paper numbers are on the full set — not a perfectly apples-to-apples comparison, but the gap is large enough to suggest the structured pipeline (plan → audit → swarm-execute in a sandbox) is doing real work beyond what a raw frontier model or simple tool use gets you.

## Quickstart

```bash
pip install -r requirements.txt

# .env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
E2B_API_KEY=...
```

Solve a problem:

```python
import asyncio
from src.core.orchestrator.orchestrate import solve_problem

result = asyncio.run(solve_problem(
    "A car accelerates from rest at 2 m/s² for 5 seconds. What is its final velocity?"
))
print(result['final_answer'], result['final_unit'])
```

Run the benchmark:

```bash
python -m src.benchmarking.cli run --config benchmark_configs/scibench_default.yaml
```

See `src/benchmarking/README.md` for the full benchmarking CLI.

## Structure

```
src/
├── core/
│   ├── router.py              # EASY/MEDIUM/HARD classifier
│   ├── single_agent/          # Fast solver for EASY/MEDIUM
│   └── orchestrator/          # Full pipeline for HARD
│       ├── orchestrate.py     # Main entry
│       ├── planner/           # Planner, Physics Lawyer, Revisor
│       ├── solver/            # K-Ahead Swarm + execution
│       ├── tools/             # Vision (18 CV tools) + E2B sandbox
│       └── prompts/
└── benchmarking/              # SciBench runner, metrics, reports
```
