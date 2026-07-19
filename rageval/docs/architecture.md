# rageval architecture

`rageval` is a **portable RAG evaluation harness** — a reusable "USB port" that measures
retrieval + answer quality and gates regressions in CI, talking to *any* RAG system through
one small adapter interface. The single rule the whole design serves:

> **The core imports nothing target-specific.** Adapters depend on the core; the core never
> depends on an adapter. That's what makes the harness `pip install`-able against any target
> and runnable end-to-end with **zero API keys** (via the deterministic MockAdapter).

## Data flow

```mermaid
flowchart TD
    subgraph targets["Targets (pluggable, behind extras)"]
        MOCK["MockAdapter<br/>(deterministic, keyless)"]
        HTTP["HTTP adapter<br/>(config-driven YAML)"]
        PY["PythonCallable adapter"]
        SB["SuperBotAdapter<br/>(M7 — POST /ask)"]
    end

    GOLDEN["Golden set<br/>(JSONL: question, reference, relevant_doc_ids)"]

    subgraph core["Core (zero target-specific imports)"]
        ADAPTER["adapter.py<br/>RAGTarget Protocol"]
        RUNNER["runner.py<br/>drive target over golden set<br/>+ tier auto-detection"]
        RESULTS["results.py<br/>RunResult / QueryRecord"]
        MANIFEST["manifest.py<br/>config+dataset+git fingerprint"]
        STORE["store.py<br/>runs/&lt;id&gt;/*.json"]
        PROVIDER["provider.py<br/>auto-from-env judge/embedder shim"]
    end

    subgraph metrics["Metrics (from scratch)"]
        RETRIEVAL["retrieval.py<br/>Recall/Precision/MRR/NDCG/HitRate"]
        ANSWER["answer.py + lexical.py<br/>token-F1/ROUGE-L, embedding, LLM-judge"]
    end

    subgraph outputs["Outputs & gate"]
        BASELINE["baseline.py<br/>snapshot + compare()"]
        TRACING["tracing.py<br/>Langfuse | local JSON"]
        REPORT["report.py<br/>static report.html"]
        DASH["dashboard.py<br/>Streamlit comparison"]
        CI["GitHub Actions<br/>rageval check → exit 1 on regression"]
    end

    MOCK & HTTP & PY & SB -. implement .-> ADAPTER
    GOLDEN --> RUNNER
    ADAPTER --> RUNNER
    RUNNER --> RESULTS
    RUNNER --> MANIFEST
    RESULTS --> STORE
    MANIFEST --> STORE
    RESULTS --> RETRIEVAL --> RESULTS
    PROVIDER --> ANSWER
    RESULTS --> ANSWER --> RESULTS
    STORE --> BASELINE
    STORE --> TRACING
    STORE --> REPORT
    STORE --> DASH
    BASELINE --> CI
```

## Why the pieces are split this way

| Component | Job | Design constraint it honors |
| --- | --- | --- |
| `core/adapter.py` | The contract every target implements (`query`, optional `retrieve`) | Core has no target imports — the seam that makes the harness portable |
| `runner.py` | Drives a target over the golden set, times it, detects the tier | Stays **metric-agnostic**: it produces a run, metrics read it afterward |
| `metrics/` | Retrieval + answer scores, implemented from first principles | Runs keyless on lexical + retrieval; judge/embedding tiers are additive |
| `core/provider.py` | Auto-from-env judge/embedder shim | Returns `None` without a key, so Mock/CI never need secrets |
| `core/manifest.py` | Config + dataset + git fingerprint per run | Makes every run reproducible and comparisons **refusable** when unfair |
| `core/baseline.py` | Snapshot a blessed run, `compare()` later runs | Only `[0,1]` quality metrics gate; different dataset → incomparable |
| `core/tracing.py` | Observability behind a `Tracer` Protocol | A tracing outage can **never** break a run's numbers |
| `report.py` / `dashboard.py` | Human-legible outputs (static HTML / interactive) | Report is pure-stdlib core; the dashboard lives behind an extra |

## The evaluation tiers

The runner adapts to whatever a target can return and records which tier ran — it never
fails because a richer tier's inputs are absent.

```mermaid
flowchart LR
    Q["target.query(question)"] --> A{"returns<br/>retrieved chunks?"}
    A -- no --> T1["tier: answer<br/>(faithfulness, relevance, latency, cost)"]
    A -- yes --> B{"golden<br/>relevant_doc_ids?"}
    B -- no --> T2["tier: answer+retrieved<br/>(+ context precision, retrieval visibility)"]
    B -- yes --> T3["tier: answer+retrieved+labelled<br/>(+ Recall/Precision/MRR/NDCG/HitRate)"]
```

## Portability check (enforced)

- `python -c "import rageval.core.adapter"` works with **no** target/host-app deps installed.
- Every optional capability (HTTP targets, tracing, dashboard, SuperBot) is a declared
  **extra**, lazily imported — so the base install stays tiny and the MockAdapter path is
  fully offline.
- CI runs lint + type-check + tests + the regression gate on every push with **no secrets**.
