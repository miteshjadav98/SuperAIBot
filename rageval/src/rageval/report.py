"""Static HTML report for a single run — self-contained, dependency-free, offline.

A run already lands on disk as ``result.json`` + ``manifest.json`` (+ ``trace.json``);
this module renders those into one ``report.html`` a human can open in a browser with no
server, no JS framework, and no network. The point of the harness is to make a regression
*legible*, so the report leads with the headline numbers and the pass/fail verdict, then
lets you drill into per-query records.

Kept deliberately in pure stdlib (``html.escape`` + string templating, inline CSS) so it
stays in the portable core: generating a report pulls in no extra and never touches a
target. The richer, interactive *comparison* view (many runs, charts) is the Streamlit
dashboard behind the ``[dashboard]`` extra — this file is the always-available artifact.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rageval.core.baseline import RegressionReport

# Metric display order mirrors the terminal tables so the report and CLI agree.
_RETRIEVAL_METRICS = ("recall", "precision", "mrr", "ndcg", "hitrate")
_ANSWER_METRIC_ORDER = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "semantic_similarity",
    "token_f1",
    "rouge_l",
)

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial,
    sans-serif;
  margin: 0; padding: 2rem 1.25rem; line-height: 1.5;
  color: #1a1d23; background: #f6f7f9;
}
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .75rem; letter-spacing: .01em; }
.sub { color: #667085; font-size: .85rem; margin: 0 0 1.5rem; }
.provenance { display: flex; flex-wrap: wrap; gap: .4rem .5rem; margin: .5rem 0 0; }
.tag {
  font-size: .72rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #eceef2; color: #475467; padding: .15rem .5rem; border-radius: 999px;
}
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .75rem; margin: 0 0 .5rem; }
.card { background: #fff; border: 1px solid #e4e7ec; border-radius: 10px; padding: .9rem 1rem; }
.card .label { color: #667085; font-size: .72rem; text-transform: uppercase;
  letter-spacing: .04em; }
.card .value { font-size: 1.5rem; font-weight: 650; margin-top: .15rem;
  font-variant-numeric: tabular-nums; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e4e7ec;
  border-radius: 10px; overflow: hidden; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .5rem .75rem; border-bottom: 1px solid #eef0f3;
  font-size: .88rem; }
th:first-child, td:first-child { text-align: left; font-family: ui-monospace, Menlo, monospace; }
thead th { background: #fafbfc; color: #475467; font-weight: 600; }
tbody tr:last-child td { border-bottom: none; }
.banner { border-radius: 10px; padding: .85rem 1.1rem; font-weight: 600; margin: 0 0 1rem; }
.banner.pass { background: #e7f6ec; color: #1a7f37; border: 1px solid #b7e4c3; }
.banner.fail { background: #fdecec; color: #b42318; border: 1px solid #f4c4c0; }
.banner.warn { background: #fef7e6; color: #935e0b; border: 1px solid #f3dfa6; }
.regressed td { color: #b42318; font-weight: 600; }
.muted { color: #98a2b3; }
.notes { margin: .25rem 0 1rem; padding-left: 1.1rem; color: #667085; font-size: .85rem; }
details { background: #fff; border: 1px solid #e4e7ec; border-radius: 10px;
  margin-bottom: .6rem; }
summary { cursor: pointer; padding: .7rem 1rem; font-weight: 550; }
summary .q { color: #1a1d23; }
.record-body { padding: 0 1rem 1rem; }
.record-body p { margin: .5rem 0; }
.record-body .k { color: #667085; font-size: .78rem; text-transform: uppercase;
  letter-spacing: .04em; }
.answer { white-space: pre-wrap; }
.chips { display: flex; flex-wrap: wrap; gap: .3rem; }
.chip { font-family: ui-monospace, Menlo, monospace; font-size: .74rem;
  background: #eef2ff; color: #3538cd; padding: .1rem .45rem; border-radius: 6px; }
.chip.hit { background: #e7f6ec; color: #1a7f37; }
footer { color: #98a2b3; font-size: .78rem; text-align: center; margin-top: 2.5rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e6e8eb; background: #14161a; }
  .sub, .card .label, thead th, .notes, .record-body .k { color: #98a2b3; }
  .tag { background: #24272e; color: #b8bec9; }
  .card, table, details { background: #1c1f24; border-color: #2a2e36; }
  th, td { border-color: #24272e; }
  thead th { background: #212429; }
  summary .q, .card .value { color: #e6e8eb; }
  .chip { background: #23263a; color: #a5b4fc; }
  .chip.hit { background: #16341f; color: #6ee7a0; }
}
"""


def _ks_from(aggregate: dict[str, float]) -> list[int]:
    """Recover the sorted cutoffs present in ``metric@k`` keys."""
    ks: set[int] = set()
    for key in aggregate:
        if "@" in key:
            suffix = key.rsplit("@", 1)[1]
            if suffix.isdigit():
                ks.add(int(suffix))
    return sorted(ks)


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _provenance(result: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Render the reproducibility fingerprint as a row of monospace tags."""
    git = manifest.get("git_sha")
    tags: list[tuple[str, str | None]] = [
        ("tier", result.get("tier")),
        ("git", git[:10] if isinstance(git, str) else None),
        ("provider", manifest.get("provider")),
        ("model", manifest.get("model")),
        ("created", manifest.get("created_at")),
    ]
    ds = manifest.get("dataset_hash")
    if isinstance(ds, str):
        tags.append(("dataset", ds[:10]))
    spans = "".join(
        f'<span class="tag">{escape(label)}: {escape(str(value))}</span>'
        for label, value in tags
        if value
    )
    return f'<div class="provenance">{spans}</div>'


def _op_cards(op: dict[str, Any]) -> str:
    cards = [
        ("queries", str(op.get("queries", 0))),
        ("latency p50", f"{_fmt(op.get('latency_p50_ms'), '.0f')} ms"),
        ("latency p95", f"{_fmt(op.get('latency_p95_ms'), '.0f')} ms"),
        (
            "total cost",
            "n/a" if op.get("total_cost_usd") is None else f"${op['total_cost_usd']:.4f}",
        ),
    ]
    body = "".join(
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div></div>'
        for label, value in cards
    )
    return f'<div class="cards">{body}</div>'


def _retrieval_table(aggregate: dict[str, float]) -> str:
    if not aggregate:
        return '<p class="muted">No labelled queries — retrieval metrics were skipped.</p>'
    ks = _ks_from(aggregate)
    head = "".join(f"<th>@{k}</th>" for k in ks)
    rows = []
    for metric in _RETRIEVAL_METRICS:
        if not any(f"{metric}@{k}" in aggregate for k in ks):
            continue
        cells = "".join(f"<td>{_fmt(aggregate.get(f'{metric}@{k}'))}</td>" for k in ks)
        rows.append(f"<tr><td>{metric}</td>{cells}</tr>")
    return (
        f"<table><thead><tr><th>metric</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _answer_table(aggregate: dict[str, float]) -> str:
    if not aggregate:
        return (
            '<p class="muted">No answer metrics — no reference answers and no '
            "judge/embedder available.</p>"
        )
    ordered = [m for m in _ANSWER_METRIC_ORDER if m in aggregate]
    ordered += sorted(m for m in aggregate if m not in _ANSWER_METRIC_ORDER)
    rows = "".join(
        f"<tr><td>{escape(m)}</td><td>{_fmt(aggregate[m])}</td></tr>" for m in ordered
    )
    return (
        "<table><thead><tr><th>answer metric</th><th>score</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _regression_section(report: RegressionReport) -> str:
    """Render the gate verdict banner + the base/curr/delta table."""
    if not report.comparable:
        banner = '<div class="banner fail">GATE: NOT COMPARABLE to baseline (see notes).</div>'
    elif report.passed:
        banner = '<div class="banner pass">GATE: PASSED — no quality regression.</div>'
    else:
        names = ", ".join(escape(d.name) for d in report.regressions)
        banner = f'<div class="banner fail">GATE: FAILED — regressions in {names}.</div>'

    notes = ""
    if report.notes:
        items = "".join(f"<li>{escape(n)}</li>" for n in report.notes)
        notes = f'<ul class="notes">{items}</ul>'

    rows = []
    for d in report.deltas + report.operational:
        cls = ' class="regressed"' if d.regressed else ""
        curr = "n/a" if d.current is None else _fmt(d.current)
        delta = "n/a" if d.delta is None else _fmt(d.delta, "+.3f")
        flag = " REGRESSED" if d.regressed else ""
        rows.append(
            f"<tr{cls}><td>{escape(d.name)}{flag}</td><td>{_fmt(d.baseline)}</td>"
            f"<td>{curr}</td><td>{delta}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>metric</th><th>base</th><th>curr</th><th>delta</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return (
        f"<h2>Regression gate (tolerance {report.tolerance:.3f})</h2>"
        f"{banner}{notes}{table}"
    )


def _record_details(records: list[dict[str, Any]]) -> str:
    """One collapsible block per query: answer, retrieved ids, and per-record metrics."""
    blocks = []
    for i, rec in enumerate(records, start=1):
        relevant = set(rec.get("relevant_doc_ids") or [])
        chips = "".join(
            f'<span class="chip{" hit" if c.get("id") in relevant else ""}">'
            f'{escape(str(c.get("id")))}</span>'
            for c in rec.get("retrieved", [])
        )
        chips_html = (
            f'<div class="chips">{chips}</div>' if chips else '<span class="muted">—</span>'
        )

        metrics = {**rec.get("retrieval", {}), **rec.get("answer_metrics", {})}
        metric_html = (
            " · ".join(f"{escape(k)}={_fmt(v)}" for k, v in sorted(metrics.items()))
            or '<span class="muted">none</span>'
        )
        ref = rec.get("reference_answer")
        ref_html = (
            f'<p><span class="k">reference</span><br>{escape(str(ref))}</p>' if ref else ""
        )
        question = escape(str(rec.get("question", "")))
        blocks.append(
            f"<details><summary><span class='q'>Q{i}. {question}</span></summary>"
            f'<div class="record-body">'
            f'<p><span class="k">answer</span><br>'
            f'<span class="answer">{escape(str(rec.get("answer", "")))}</span></p>'
            f"{ref_html}"
            f'<p><span class="k">retrieved</span></p>{chips_html}'
            f'<p><span class="k">metrics</span><br>{metric_html}</p>'
            f"</div></details>"
        )
    return "".join(blocks)


def render_report(
    result: dict[str, Any],
    manifest: dict[str, Any],
    *,
    regression: RegressionReport | None = None,
) -> str:
    """Render a run (and optional regression verdict) as a self-contained HTML document."""
    op = result.get("operational", {})
    run_id = escape(str(result.get("run_id", "unknown")))
    target = escape(str(result.get("target_name", "unknown")))

    regression_html = _regression_section(regression) if regression is not None else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rageval report — {run_id}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>rageval report</h1>
  <p class="sub">target <strong>{target}</strong> · run <code>{run_id}</code></p>
  {_provenance(result, manifest)}

  {regression_html}

  <h2>Operational</h2>
  {_op_cards(op)}

  <h2>Retrieval metrics</h2>
  {_retrieval_table(result.get("retrieval_aggregate", {}))}

  <h2>Answer metrics</h2>
  {_answer_table(result.get("answer_aggregate", {}))}

  <h2>Per-query records</h2>
  {_record_details(result.get("records", []))}

  <footer>Generated by rageval · portable RAG evaluation harness · this file is
  fully self-contained (no network, no scripts).</footer>
</main>
</body>
</html>
"""
