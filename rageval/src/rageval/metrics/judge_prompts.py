"""Versioned judge prompts — the exact text sent to the LLM judge, pinned in-repo.

Judge-based metrics are only comparable across runs if the judge *prompt* is stable, so
every prompt here carries a version that is recorded into each run. Changing a prompt
means bumping its version — a silent wording tweak that shifts scores would otherwise
masquerade as a quality regression (or improvement) in the target under test.

All prompts demand a single strict-JSON object reply. The parser in ``answer.py`` is
deliberately unforgiving: if the judge doesn't produce a parsable verdict, the metric is
*skipped* for that record — never defaulted to a fake neutral score. (This is a lesson
taken from a production system we reviewed that defaulted unparsable judge replies to
0.5, silently converting judge failures into plausible-looking data.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JudgePrompt:
    """A named, versioned prompt template. ``template`` uses ``str.format`` fields."""

    name: str
    version: str
    template: str

    def render(self, **fields: str) -> str:
        return self.template.format(**fields)


FAITHFULNESS_V1 = JudgePrompt(
    name="faithfulness",
    version="1.0",
    template="""\
You are a strict evaluator checking whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Score how faithful the answer is to the context on a 0.0-1.0 scale:
- 1.0: every factual claim in the answer is directly supported by the context.
- 0.5: some claims are supported, others are unsupported or embellished.
- 0.0: the answer contradicts the context or is unrelated to it.
Do not reward fluency; only grounding matters. If the answer declines to answer
("I don't know"), score 1.0 only when the context indeed lacks the information.

Reply with ONLY a JSON object, no prose: {{"score": <float 0-1>, "reason": "<one sentence>"}}""",
)

ANSWER_RELEVANCE_V1 = JudgePrompt(
    name="answer_relevance",
    version="1.0",
    template="""\
You are a strict evaluator checking whether an answer actually addresses the question.

Question:
{question}

Answer:
{answer}

Score relevance on a 0.0-1.0 scale:
- 1.0: directly and completely addresses the question asked.
- 0.5: partially addresses it, or answers a related-but-different question.
- 0.0: does not address the question at all (including generic deflections).
Ignore whether the answer is factually correct; only relevance to the question matters.

Reply with ONLY a JSON object, no prose: {{"score": <float 0-1>, "reason": "<one sentence>"}}""",
)

CONTEXT_PRECISION_V1 = JudgePrompt(
    name="context_precision",
    version="1.0",
    template="""\
You are evaluating a retrieval system. For the question below, decide which of the
numbered context chunks are actually useful for answering it.

Question:
{question}

Chunks:
{chunks}

Reply with ONLY a JSON object, no prose: {{"relevant_indices": [<0-based ints>]}}
An empty list is a valid reply if no chunk is useful.""",
)

CONTEXT_RECALL_V1 = JudgePrompt(
    name="context_recall",
    version="1.0",
    template="""\
You are evaluating whether retrieved context contains the information needed to produce
a reference answer.

Reference answer:
{reference}

Retrieved context:
{context}

Break the reference answer into its factual statements and count how many of them are
supported by the retrieved context.

Reply with ONLY a JSON object, no prose:
{{"supported": <int>, "total": <int>, "reason": "<one sentence>"}}""",
)

# The registry recorded into every scored run: metric name -> prompt version.
PROMPT_VERSIONS: dict[str, str] = {
    p.name: p.version
    for p in (FAITHFULNESS_V1, ANSWER_RELEVANCE_V1, CONTEXT_PRECISION_V1, CONTEXT_RECALL_V1)
}
