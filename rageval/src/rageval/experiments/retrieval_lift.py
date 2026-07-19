"""Prove-a-lift — the harness measuring a *real* retrieval improvement, keyless.

The whole pitch of the harness is that it can tell you whether a change to a RAG system
made retrieval better or worse. This experiment demonstrates exactly that on a self-contained
IT-support knowledge base: it scores two retrievers with the harness's own metric layer and
reports the before/after delta.

  * **baseline** — BM25 only (exact-token lexical scoring).
  * **hybrid**   — Reciprocal-Rank-Fusion of BM25 with a character-n-gram cosine signal.

That's the same architecture SuperBot's ``_hybrid_retrieve`` uses (RRF fusing a lexical and
a second signal), reproduced here in pure stdlib so the experiment is deterministic and runs
with no API keys — you can re-run it and get the same numbers, and CI can guard the claim.

Why hybrid wins here (and in the real world): BM25 only matches *exact* tokens, so a query
like "reset my password" scores a doc titled "Password complexity policy" (shares the exact
word "password") above the actually-relevant "Resetting forgotten passwords" (which uses the
inflected forms "resetting"/"passwords"). The character-n-gram signal sees the shared
sub-word structure BM25 misses, and RRF fusion pulls the right doc back into the top-k. The
n-gram cosine is a deliberately lightweight, keyless stand-in for a real embedding model —
swap in a true embedder in production; the *fusion* is the point.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import mean

from rageval.core.adapter import RetrievedChunk
from rageval.core.results import QueryRecord, RunResult
from rageval.datasets.golden import GoldenRecord
from rageval.metrics.evaluate import score_run

Retriever = Callable[[str], list[RetrievedChunk]]

# --- the knowledge base -------------------------------------------------------------
# The relevant doc for each "hard" query phrases the topic with *inflected* forms only
# (resetting/passwords/logins) so exact-token BM25 scores it near zero, while a distractor
# shares one exact query token (e.g. "password") and so wins under BM25. The character-
# n-gram signal sees the shared sub-word structure across several terms and ranks the real
# doc first; RRF fusion then lifts it into the top-k. "Easy" queries share an exact token
# with their relevant doc, so BM25 already succeeds — they show hybrid doesn't regress.
CORPUS: dict[str, str] = {
    # hard-query targets (inflections only) + their exact-token distractors
    "kb-pwd-reset": "Resetting passwords and logins from the self-service portal.",
    "kb-pwd-policy": "Password strength policy, expiry and reuse rules.",
    "kb-battery": "Batteries draining fast: charging cycles and calibration.",
    "kb-power-plan": "Battery saver and power plan options on Windows.",
    "kb-account-lock": "Unlocking locked accounts after repeated failed logins.",
    "kb-account-new": "Creating a user account for a new hire.",
    "kb-admin-rights": "Granting administrator permissions to a user temporarily.",
    "kb-admin-audit": "Admin action audit log export and review.",
    # easy-query targets (share an exact token with the query)
    "kb-printer": "Printer driver installation and updates on laptops.",
    "kb-update-fail": "Software update and patch rollback troubleshooting.",
    "kb-vpn": "Corporate VPN client connection guide for remote work.",
    "kb-email-mobile": "Configuring email on phones and tablets with the company app.",
    # distractor filler to give the corpus depth
    "kb-wifi": "Fixing flaky office wifi: forget the network and rejoin.",
    "kb-storage": "Freeing disk storage by clearing temporary files.",
    "kb-mfa": "Enrolling a new device for multi factor authentication.",
    "kb-monitor": "Connecting an external monitor and display scaling.",
}

# Queries in natural base forms; each labelled with the single relevant doc id.
# The first four are "hard" (BM25's exact match lands on a distractor); the rest are "easy".
GOLDEN: list[GoldenRecord] = [
    GoldenRecord("reset password login", ["kb-pwd-reset"], None),
    GoldenRecord("drain battery charge", ["kb-battery"], None),
    GoldenRecord("unlock account login", ["kb-account-lock"], None),
    GoldenRecord("grant admin permission", ["kb-admin-rights"], None),
    GoldenRecord("install printer driver", ["kb-printer"], None),
    GoldenRecord("update software patch", ["kb-update-fail"], None),
    GoldenRecord("connect vpn", ["kb-vpn"], None),
    GoldenRecord("configure email phone", ["kb-email-mobile"], None),
]

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# --- signal 1: BM25 (exact-token lexical) -------------------------------------------
class BM25:
    """Textbook BM25 over a small corpus, implemented from scratch (k1=1.5, b=0.75)."""

    def __init__(self, corpus: dict[str, str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids = list(corpus)
        self._tf: dict[str, Counter[str]] = {i: Counter(_tokenize(t)) for i, t in corpus.items()}
        self._len = {i: sum(tf.values()) for i, tf in self._tf.items()}
        self._avglen = mean(self._len.values()) if self._len else 0.0
        n = len(self.ids)
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        terms = _tokenize(query)
        out: dict[str, float] = {}
        for doc_id in self.ids:
            tf = self._tf[doc_id]
            length = self._len[doc_id]
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * length / self._avglen)
                score += idf * (freq * (self.k1 + 1)) / denom
            out[doc_id] = score
        return out


# --- signal 2: character-n-gram cosine (keyless sub-word similarity) -----------------
def _char_ngrams(text: str, n: int = 3) -> Counter[str]:
    grams: Counter[str] = Counter()
    for word in _tokenize(text):
        padded = f"#{word}#"
        for i in range(len(padded) - n + 1):
            grams[padded[i : i + n]] += 1
    return grams


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(count * b[gram] for gram, count in a.items() if gram in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class CharCosine:
    """Sub-word similarity: cosine over character-trigram bags. Catches inflections."""

    def __init__(self, corpus: dict[str, str], *, n: int = 3) -> None:
        self.n = n
        self._vecs = {i: _char_ngrams(t, n) for i, t in corpus.items()}

    def scores(self, query: str) -> dict[str, float]:
        qv = _char_ngrams(query, self.n)
        return {doc_id: _cosine(qv, vec) for doc_id, vec in self._vecs.items()}


# --- ranking + fusion ---------------------------------------------------------------
def _rank(scores: dict[str, float]) -> list[str]:
    """Doc ids best-first; ties broken by id so the ordering is fully deterministic."""
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _rrf(rankings: list[list[str]], *, k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion — combine several rankings into one (higher = better)."""
    agg: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            agg[doc_id] = agg.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return _rank(agg)


def make_retrievers(
    corpus: dict[str, str], *, top_k: int = 5
) -> tuple[Retriever, Retriever]:
    """Build the (baseline BM25-only, hybrid RRF) retriever pair over ``corpus``."""
    bm25 = BM25(corpus)
    charcos = CharCosine(corpus)

    def _chunks(ids: list[str]) -> list[RetrievedChunk]:
        return [RetrievedChunk(id=i, text=corpus[i], score=None) for i in ids]

    def baseline(question: str) -> list[RetrievedChunk]:
        return _chunks(_rank(bm25.scores(question))[:top_k])

    def hybrid(question: str) -> list[RetrievedChunk]:
        fused = _rrf([_rank(bm25.scores(question)), _rank(charcos.scores(question))])
        return _chunks(fused[:top_k])

    return baseline, hybrid


# --- evaluation via the harness's own metric layer ----------------------------------
def evaluate(name: str, retriever: Retriever, golden: list[GoldenRecord]) -> RunResult:
    """Run a retriever over the golden set and score it with the harness's ``score_run``."""
    records = [
        QueryRecord(
            question=g.question,
            answer="",
            retrieved=list(retriever(g.question)),
            relevant_doc_ids=list(g.relevant_doc_ids),
        )
        for g in golden
    ]
    result = RunResult(
        run_id=f"lift-{name}",
        target_name=name,
        tier="answer+retrieved+labelled",
        records=records,
    )
    score_run(result)
    return result


# Metrics we headline in the delta table (the ones a retrieval change should move).
_HEADLINE = ("recall@3", "mrr@3", "ndcg@3", "recall@1", "hitrate@3")


def compare(golden: list[GoldenRecord] | None = None) -> dict[str, dict[str, float]]:
    """Return ``{metric: {baseline, hybrid, delta}}`` for the headline retrieval metrics."""
    gold = golden if golden is not None else GOLDEN
    baseline_fn, hybrid_fn = make_retrievers(CORPUS)
    base = evaluate("baseline-bm25", baseline_fn, gold).retrieval_aggregate
    hyb = evaluate("hybrid-rrf", hybrid_fn, gold).retrieval_aggregate
    table: dict[str, dict[str, float]] = {}
    for metric in _HEADLINE:
        b = base.get(metric, 0.0)
        h = hyb.get(metric, 0.0)
        table[metric] = {"baseline": b, "hybrid": h, "delta": h - b}
    return table


def format_table(table: dict[str, dict[str, float]]) -> str:
    header = "metric".ljust(12) + "baseline".rjust(10) + "hybrid".rjust(10) + "delta".rjust(10)
    lines = [header, "-" * len(header)]
    for metric, row in table.items():
        lines.append(
            metric.ljust(12)
            + f"{row['baseline']:10.3f}{row['hybrid']:10.3f}{row['delta']:+10.3f}"
        )
    return "\n".join(lines)


def _render_markdown(table: dict[str, dict[str, float]]) -> str:
    rows = "\n".join(
        f"| {m} | {r['baseline']:.3f} | {r['hybrid']:.3f} | {r['delta']:+.3f} |"
        for m, r in table.items()
    )
    return (
        "# Prove-a-lift: RRF hybrid vs BM25-only retrieval\n\n"
        "Generated by `python -m rageval.experiments.retrieval_lift`. Deterministic and "
        "keyless — re-running reproduces these exact numbers.\n\n"
        f"Corpus: {len(CORPUS)} docs. Golden: {len(GOLDEN)} labelled queries.\n\n"
        "| metric | baseline (BM25) | hybrid (RRF) | delta |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "Hybrid retrieval fuses BM25 with a character-n-gram cosine signal via Reciprocal "
        "Rank Fusion, recovering relevant documents that exact-token BM25 ranks below "
        "distractors. The harness's own retrieval metrics quantify the improvement.\n"
    )


def main() -> None:
    """Print the before/after table and write ``RESULTS.md`` next to this experiment."""
    table = compare()
    print("Prove-a-lift: RRF hybrid vs BM25-only retrieval\n")
    print(format_table(table))
    out = Path(__file__).with_name("RESULTS.md")
    out.write_text(_render_markdown(table), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
