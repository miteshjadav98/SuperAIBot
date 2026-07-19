# rageval — a portable RAG evaluation harness

> **Plug any RAG / ask / chat API in, get retrieval + answer + cost metrics and a CI
> regression gate.** The harness is a reusable "USB port," not welded to one app: it
> talks to any target through a tiny adapter interface, and its core imports nothing
> target-specific. This is a RAG-eval SDK with pluggable targets — like a test runner
> that's independent of the app under test.

> ⚠️ **Status: M5.** Shipped so far: the adapter contract, result types, run manifest and
> store, the offline **MockAdapter**, the runner, retrieval metrics, the config-driven
> HTTP + PythonCallable adapters, answer metrics (lexical / embedding / LLM-judge), and —
> this milestone — **run tracing and a baseline regression gate wired into CI**. The
> reports and dashboard land in M6. The whole harness still runs with **zero API keys**.

## The adapter contract (the "USB port")

Every target — SuperBot, a bare `/chat` endpoint, a local retriever — implements one
small interface. The core depends on adapters *never*; adapters depend on the core.

```python
@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float | None = None

@dataclass
class RAGResult:
    answer: str
    retrieved: list[RetrievedChunk]   # may be empty if the target is answer-only
    latency_ms: float
    cost_usd: float | None = None
    raw: dict | None = None

class RAGTarget(Protocol):
    name: str
    def query(self, question: str, **context) -> RAGResult: ...
    def retrieve(self, question: str, k: int, **context) -> list[RetrievedChunk]: ...  # optional
```

**Write your own adapter in ~20 lines** — wrap any API and you're done:

```python
import httpx
from rageval.core.adapter import RAGResult, RetrievedChunk

class MyAdapter:
    name = "my-api"
    def query(self, question: str, **ctx) -> RAGResult:
        r = httpx.post("https://my.api/ask", json={"q": question}, timeout=30).json()
        chunks = [RetrievedChunk(id=s["id"], text=s["text"], score=s.get("score"))
                  for s in r.get("sources", [])]
        return RAGResult(answer=r["answer"], retrieved=chunks, latency_ms=0.0)
```

That's the whole integration cost. (A **config-driven HTTP adapter** — no code, just YAML
field mappings — arrives in M3, so most APIs won't even need this.)

## Tiered evaluation (the headline feature)

The harness adapts to whatever a target can give and prints which tier ran — it never
fails because a richer tier's inputs are absent:

| Target provides | Metrics produced |
| --- | --- |
| answer only (e.g. bare `/chat`) | faithfulness, answer relevance, latency, cost |
| answer + retrieved chunks (no labels) | above + context precision (self-eval), retrieval visibility |
| answer + retrieved + **golden set** | above + Recall / Precision / MRR / NDCG / HitRate |

*(Retrieval metrics M2 ✅; answer metrics M4 ✅ — lexical token-F1/ROUGE-L for free,
embedding similarity and LLM-judge faithfulness/relevance/context when a key is present.
The tier is detected per run.)*

## Quickstart (< 5 minutes, no keys)

```bash
cd rageval
pip install -e ".[dev]"

# Run the offline MockAdapter over the tiny golden set and score retrieval:
rageval run --golden data/golden/tiny.jsonl
# → run_id=... tier=answer+retrieved+labelled queries=3
#   latency p50=...ms p95=...ms cost=$0.0
#
#   metric           @1      @3      @5
#   -----------------------------------
#   recall        0.333   0.333   0.667
#   precision     0.333   0.111   0.133
#   mrr           0.333   0.333   0.400
#   ndcg          0.333   0.333   0.462
#   hitrate       0.333   0.333   0.667

pytest        # green with zero API keys / network
```

Each run writes `runs/<run_id>/result.json` (per-query records + a cost/latency table),
`manifest.json` (config hash, git SHA, dataset hash, tier, judge prompt versions), and
`trace.json` (one span per query) for reproducibility.

## Regression gate (fail CI when quality drops)

The point of the harness is to **catch a quality regression before it ships**. You bless a
run you've inspected as the baseline, then gate later runs against it:

```bash
# 1. Run, inspect, and snapshot the numbers you accept as the reference:
rageval run --golden data/golden/tiny.jsonl --runs-dir runs
rageval baseline <run_id> --out baseline.json

# 2. In CI (or before a merge), re-run and fail if quality regressed:
rageval check --golden data/golden/tiny.jsonl --baseline baseline.json --no-judge
# → prints a per-metric base/curr/delta table, then:
#   gate: PASSED - no quality regression against baseline.   (exit 0)
#   gate: FAILED - regressions in: recall@5, token_f1        (exit 1)
```

Design choices that make the gate trustworthy:

- **Only quality metrics gate.** Retrieval + answer metrics live in `[0, 1]`, so an
  absolute `--tolerance` (default `0.01`) is meaningful. Latency and cost are shown as
  deltas but never fail CI — wall-clock time is environment-dependent and would flake.
- **A missing metric is a note, not a failure.** A keyless CI run can't reproduce the
  LLM-judge metrics a locally-keyed baseline recorded, so their absence is reported, not
  punished. The gate holds the line on what the run could actually measure.
- **Apples-to-oranges is refused.** A different dataset hash makes the runs incomparable
  and fails safe; a tier or judge-prompt-version change is surfaced as a note.

A committed `baseline.json` (from the deterministic MockAdapter) plus
[`.github/workflows/eval.yml`](.github/workflows/eval.yml) means the gate runs on every
push with **no secrets** — lint, type-check, tests, and the regression check.

## Tracing

Every run is traced behind a tiny `Tracer` interface that degrades gracefully and never
fails a run. With no keys you get a local `trace.json` (one span per query: input, output,
latency, retrieved ids, and the scored metrics). Set `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` (extra: `pip install 'rageval[tracing]'`) and the same runs stream to
Langfuse instead — a tracing outage still can't break the eval.

## Package layout

```
rageval/
├── src/rageval/
│   ├── core/         # adapter contract, result types, manifest, store — ZERO target deps
│   ├── adapters/     # mock (here now); http, superbot (later, behind extras)
│   ├── datasets/     # golden-set loaders
│   ├── runner.py     # drives a target over a golden set → RunResult
│   └── cli.py        # `rageval` command
├── configs/          # example target configs
├── data/golden/      # versioned golden JSONL (hashed into each run manifest)
└── tests/            # run with no keys / no network
```

## Roadmap

- **M1 ✅** Core contract + MockAdapter + runner.
- **M2 ✅** Retrieval metrics from scratch (Recall/Precision/MRR/NDCG/HitRate) + hand-checked fixtures.
- **M3 ✅** Config-driven HTTP adapter + PythonCallable adapter + tier auto-detection.
- **M4 ✅** Answer metrics (lexical, embedding, LLM-judge faithfulness/relevance/context) + versioned judge prompts.
- **M5 ✅** Run tracing (Langfuse + local-JSON fallback), baseline save/check regression gate, GitHub Actions CI (this milestone).
- **M6** Static HTML report, Streamlit dashboard, architecture diagram, full README.
- **M7** `SuperBotAdapter` — proof the contract works on a real app (`POST /ask`).
- **M8** Prove-a-lift: hybrid + reranked retrieval, before/after metric delta.

## Design notes

- **Portable by construction.** The core has no target-specific imports and its own tiny
  provider shim (`core/provider.py`), so it never depends on the host app's LLM factory.
- **Runs keyless.** MockAdapter is deterministic and offline, so CI and a fresh clone work
  with no secrets — a hard requirement, not a convenience.
- **Reproducible.** Every run writes a manifest fingerprint, and the regression gate
  refuses apples-to-oranges comparisons (different dataset hash → incomparable, fail-safe).
