# rageval — a portable RAG evaluation harness

> **Plug any RAG / ask / chat API in, get retrieval + answer + cost metrics and a CI
> regression gate.** The harness is a reusable "USB port," not welded to one app: it
> talks to any target through a tiny adapter interface, and its core imports nothing
> target-specific. This is a RAG-eval SDK with pluggable targets — like a test runner
> that's independent of the app under test.

> ✅ **Status: M8 — feature-complete.** Shipped: the adapter contract, result types, run
> manifest and store, the offline **MockAdapter**, the runner, retrieval metrics, the
> config-driven HTTP + PythonCallable adapters, answer metrics (lexical / embedding /
> LLM-judge), run tracing and a baseline regression gate wired into CI, a self-contained
> HTML report + a Streamlit dashboard + an [architecture diagram](docs/architecture.md), the
> **`SuperBotAdapter`** (first adapter against a real running app — verified live against
> `POST /ask`), and — this milestone — a **prove-a-lift experiment**: the harness measuring
> a real retrieval improvement (RRF hybrid vs BM25). The core still runs with **zero API
> keys**.

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

## Reports & dashboard

Two ways to read a run — one always-available, one interactive:

```bash
# A self-contained HTML report for a stored run (no server, no network, no scripts).
# Pass --baseline to fold the regression verdict + per-metric deltas into the page.
rageval report <run_id> --baseline baseline.json
# → report written to runs/<run_id>/report.html   (open it in any browser)

# The interactive comparison view — line up several runs and watch a metric move.
pip install -e ".[dashboard]"
rageval dashboard          # → http://localhost:8501, reads the same runs/ directory
```

The **report** is pure stdlib (it lives in the portable core, so generating one pulls in no
extra and never touches a target); the **dashboard** is Streamlit behind the `[dashboard]`
extra. Both read the exact `runs/<id>/*.json` the CLI writes — the terminal tables, the HTML
report, and the dashboard are three views of one source of truth.

See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram and the
one rule the design serves: **the core imports nothing target-specific.**

## Evaluating a real app: the SuperBot adapter

M1–M6 proved the harness on the MockAdapter and canned HTTP responses; the `SuperBotAdapter`
(extra: `[superbot]`) is the first adapter against a **live** RAG service — this repo's own
SuperBot gateway — and it demonstrates both evaluation modes:

```bash
pip install -e ".[superbot]"

# Credentials come from the environment, never the config (nothing secret is hashed into
# the manifest). Provide a token, or an email + password to log in with:
export SUPERBOT_EMAIL=you@example.com SUPERBOT_PASSWORD=...

# Black-box: log in, POST /ask, score the answer. /ask returns only {answer}, so this is
# an answer-tier run (answer + operational metrics; retrieval is skipped, by design).
rageval run --golden data/golden/tiny.jsonl --target configs/superbot.yaml
```

- **Black-box** is the honest default: a deployed API that returns only an answer *can only*
  be scored on answer quality, and the tier logic says so rather than inventing retrieval
  numbers.
- **White-box** (`white_box: true` in the config) additionally scores retrieval by calling
  the app's own `_hybrid_retrieve` directly — unlocking Recall / Precision / MRR / NDCG.
  Chunks are scored at **document granularity** (a golden `relevant_doc_ids` entry is a
  source PDF filename), since SuperBot chunks carry a `source` but no per-chunk id.

Crucially, **the harness core still never imports SuperBot**: the adapter lives behind an
extra, `config.build_target` imports it lazily, and the white-box retriever is imported
*inside a method* — so a plain `pip install rageval` needs none of the app.

## Prove-a-lift: the harness measuring a real improvement

The point of an eval harness is to answer *"did this change make retrieval better?"* with a
number. The bundled experiment does exactly that — deterministically and with no API keys:

```bash
python -m rageval.experiments.retrieval_lift
```

It scores two retrievers over a small IT-support knowledge base with the harness's own
metric layer: a **BM25-only** baseline vs a **hybrid** that fuses BM25 with a
character-n-gram signal via Reciprocal Rank Fusion (the same fusion architecture SuperBot's
`_hybrid_retrieve` uses). The measured lift:

| metric | baseline (BM25) | hybrid (RRF) | delta |
| --- | --- | --- | --- |
| recall@3 | 0.625 | 0.875 | **+0.250** |
| mrr@3 | 0.562 | 0.750 | **+0.188** |
| ndcg@3 | 0.579 | 0.783 | **+0.204** |
| recall@1 | 0.500 | 0.625 | +0.125 |
| hitrate@3 | 0.625 | 0.875 | +0.250 |

BM25 alone matches only *exact* tokens, so a query like "reset password login" scores the
distractor "Password strength policy" (shares the exact word "password") above the relevant
"Resetting passwords and logins" (which only uses inflected forms). The sub-word signal sees
the shared structure across several terms and RRF fusion pulls the right doc back into the
top-3 — recovering three of the four queries BM25 gets wrong. The numbers are
[regression-tested](tests/test_retrieval_lift.py) so the claim can't silently rot, and the
full write-up regenerates to [`RESULTS.md`](src/rageval/experiments/RESULTS.md). *(The
n-gram cosine is a keyless stand-in for a real embedding model — swap one in for production;
the fusion is what the harness is proving out.)*

## Package layout

```
rageval/
├── src/rageval/
│   ├── core/         # adapter contract, results, manifest, store, baseline, tracing — ZERO target deps
│   ├── adapters/     # mock, http, python_callable; superbot (behind the [superbot] extra)
│   ├── metrics/      # retrieval + answer metrics, implemented from scratch
│   ├── datasets/     # golden-set loaders
│   ├── experiments/  # reproducible keyless demos (prove-a-lift retrieval experiment)
│   ├── runner.py     # drives a target over a golden set → RunResult
│   ├── report.py     # self-contained static HTML report (pure stdlib)
│   ├── dashboard.py  # Streamlit comparison view (behind the [dashboard] extra)
│   └── cli.py        # `rageval` command
├── configs/          # example target configs
├── data/golden/      # versioned golden JSONL (hashed into each run manifest)
├── docs/             # architecture diagram + design notes
└── tests/            # run with no keys / no network
```

## Roadmap

- **M1 ✅** Core contract + MockAdapter + runner.
- **M2 ✅** Retrieval metrics from scratch (Recall/Precision/MRR/NDCG/HitRate) + hand-checked fixtures.
- **M3 ✅** Config-driven HTTP adapter + PythonCallable adapter + tier auto-detection.
- **M4 ✅** Answer metrics (lexical, embedding, LLM-judge faithfulness/relevance/context) + versioned judge prompts.
- **M5 ✅** Run tracing (Langfuse + local-JSON fallback), baseline save/check regression gate, GitHub Actions CI (this milestone).
- **M6 ✅** Static HTML report (self-contained), Streamlit comparison dashboard, architecture diagram.
- **M7 ✅** `SuperBotAdapter` — the contract on a real app: black-box `POST /ask` + opt-in white-box retrieval.
- **M8 ✅** Prove-a-lift: RRF hybrid vs BM25 retrieval, harness-measured before/after delta (recall@3 +0.25).

## Design notes

- **Portable by construction.** The core has no target-specific imports and its own tiny
  provider shim (`core/provider.py`), so it never depends on the host app's LLM factory.
- **Runs keyless.** MockAdapter is deterministic and offline, so CI and a fresh clone work
  with no secrets — a hard requirement, not a convenience.
- **Reproducible.** Every run writes a manifest fingerprint, and the regression gate
  refuses apples-to-oranges comparisons (different dataset hash → incomparable, fail-safe).
