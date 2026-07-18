"""Lexical answer metrics — judge-free, key-free, from first principles.

These compare a generated answer against a golden ``reference_answer`` using surface
overlap only. They are cheap, deterministic, and run in CI with zero API keys — the
answer-metric floor beneath the LLM-judge metrics, exactly as the retrieval metrics are
the floor for retrieval. Both are blind to paraphrase (that's what the judge and the
embedding-similarity metrics are for); their value is being un-gameable and free.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens; punctuation-insensitive so formatting can't move scores."""
    return _TOKEN_RE.findall(text.lower())


def token_f1(answer: str, reference: str) -> float:
    """Token-level F1 between answer and reference (the classic SQuAD metric).

    Formula: precision = |overlap| / |answer tokens|, recall = |overlap| / |reference
    tokens|, F1 = 2PR/(P+R), where overlap counts tokens with multiplicity (bag
    intersection).
    Intuition: rewards using the reference's words without demanding exact order — robust
    for short factual answers ("Paris" vs "It is Paris.").
    Failure mode: blind to word order and to paraphrase — "dog bites man" scores 1.0
    against "man bites dog", and a correct synonym scores 0. Pair with semantic metrics.
    """
    a, r = _tokens(answer), _tokens(reference)
    if not a or not r:
        return 1.0 if a == r else 0.0
    # Bag intersection: count shared tokens with multiplicity.
    counts: dict[str, int] = {}
    for tok in a:
        counts[tok] = counts.get(tok, 0) + 1
    overlap = 0
    for tok in r:
        if counts.get(tok, 0) > 0:
            counts[tok] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def rouge_l(answer: str, reference: str) -> float:
    """ROUGE-L F-measure: longest-common-subsequence overlap between answer and reference.

    Formula: with L = LCS(answer, reference) over tokens, P = L/|answer|, R = L/|reference|,
    score = 2PR/(P+R).
    Intuition: unlike token-F1, the LCS respects *order* — credit is only given to tokens
    appearing in the same sequence as the reference, so it rewards fluent restatements
    over keyword soup.
    Failure mode: still lexical (paraphrase scores 0), and O(len_a * len_r) — fine for
    answers, wrong tool for whole documents.
    """
    a, r = _tokens(answer), _tokens(reference)
    if not a or not r:
        return 1.0 if a == r else 0.0
    # Classic DP over token sequences, rolling one row to keep memory at O(len_r).
    prev = [0] * (len(r) + 1)
    for tok_a in a:
        curr = [0] * (len(r) + 1)
        for j, tok_r in enumerate(r, start=1):
            curr[j] = prev[j - 1] + 1 if tok_a == tok_r else max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[len(r)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(a)
    recall = lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(u: list[float], v: list[float]) -> float:
    """Cosine similarity between two vectors (helper for embedding-based metrics).

    Kept here (stdlib-only) rather than importing numpy/sklearn: two short vectors per
    call don't justify a dependency, and the harness core stays tiny.
    """
    if len(u) != len(v) or not u:
        raise ValueError("vectors must be non-empty and the same length")
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    # ``** 0.5`` is typed as returning Any (it may yield complex), so name the norms as
    # floats explicitly to keep the function's return type honest.
    norm_u: float = sum(x * x for x in u) ** 0.5
    norm_v: float = sum(y * y for y in v) ** 0.5
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot / (norm_u * norm_v)
