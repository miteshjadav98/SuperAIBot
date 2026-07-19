"""Streamlit comparison dashboard — browse and diff runs (extra: ``[dashboard]``).

This is the *interactive* counterpart to the static ``report.html``: it reads the same
``runs/`` directory the CLI writes and lets you line up several runs to see how a metric
moved over time. It's deliberately kept out of the core — Streamlit lives behind the
``[dashboard]`` extra, and nothing in the harness imports this module except the
``rageval dashboard`` launcher, which shells out to ``streamlit run`` on this file.

Run it via ``rageval dashboard`` (preferred) or point Streamlit at this file directly::

    streamlit run src/rageval/dashboard.py -- --runs-dir runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import streamlit as st

# Metric families, in the same display order as the terminal tables and HTML report.
_RETRIEVAL = ("recall", "precision", "mrr", "ndcg", "hitrate")
_ANSWER_ORDER = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "semantic_similarity",
    "token_f1",
    "rouge_l",
)


def _runs_dir() -> Path:
    """Resolve --runs-dir from the args Streamlit forwards after the ``--`` separator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    args, _ = parser.parse_known_args()
    return Path(args.runs_dir)


def load_runs(root: Path) -> list[dict[str, Any]]:
    """Load every stored run under ``root`` as ``{result, manifest, run_id, created_at}``.

    Newest first (by manifest ``created_at``), and resilient: a half-written or malformed
    run directory is skipped rather than crashing the whole dashboard.
    """
    runs: list[dict[str, Any]] = []
    if not root.exists():
        return runs
    for entry in sorted(root.iterdir()):
        result_file = entry / "result.json"
        manifest_file = entry / "manifest.json"
        if not (result_file.exists() and manifest_file.exists()):
            continue
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            {
                "run_id": result.get("run_id", entry.name),
                "created_at": manifest.get("created_at", ""),
                "result": result,
                "manifest": manifest,
            }
        )
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


def _quality_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(result.get("retrieval_aggregate", {}))
    metrics.update(result.get("answer_aggregate", {}))
    return metrics


def _metric_sort_key(name: str) -> tuple[int, int, str]:
    """Order metrics retrieval-first (by family then k), then known answer metrics."""
    base, _, k = name.partition("@")
    if base in _RETRIEVAL:
        return (0, _RETRIEVAL.index(base) * 100 + (int(k) if k.isdigit() else 0), name)
    if name in _ANSWER_ORDER:
        return (1, _ANSWER_ORDER.index(name), name)
    return (2, 0, name)


def _label(run: dict[str, Any]) -> str:
    created = str(run["created_at"])[:19].replace("T", " ")
    return f"{run['run_id']}  ({run['result'].get('tier', '?')}, {created})"


def render(runs: list[dict[str, Any]]) -> None:
    """Draw the whole dashboard for the given runs. Kept side-effecting on ``st``."""
    st.set_page_config(page_title="rageval dashboard", page_icon="*", layout="wide")
    st.title("rageval — run comparison")
    st.caption("Interactive view over the same runs/ directory the CLI writes.")

    if not runs:
        st.info("No runs found. Produce one with `rageval run --golden ...`.")
        return

    by_label = {_label(r): r for r in runs}
    default = list(by_label)[: min(3, len(by_label))]
    chosen = st.multiselect("Runs to compare", list(by_label), default=default)
    selected = [by_label[label] for label in chosen]
    if not selected:
        st.warning("Select at least one run.")
        return

    # --- side-by-side quality metrics -------------------------------------------------
    st.subheader("Quality metrics")
    metric_names = sorted(
        {name for r in selected for name in _quality_metrics(r["result"])},
        key=_metric_sort_key,
    )
    if metric_names:
        table: list[dict[str, Any]] = []
        for name in metric_names:
            row: dict[str, Any] = {"metric": name}
            for r in selected:
                value = _quality_metrics(r["result"]).get(name)
                row[r["run_id"]] = None if value is None else round(value, 3)
            table.append(row)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.bar_chart(
            {
                r["run_id"]: [
                    _quality_metrics(r["result"]).get(name, 0.0) for name in metric_names
                ]
                for r in selected
            }
        )
    else:
        st.write("No quality metrics in the selected runs.")

    # --- operational rollups ----------------------------------------------------------
    st.subheader("Operational")
    op_cols = st.columns(len(selected))
    for col, r in zip(op_cols, selected, strict=False):
        op = r["result"].get("operational", {})
        with col:
            st.metric("run", r["run_id"])
            st.write(f"queries: {op.get('queries', 0)}")
            st.write(f"latency p50: {op.get('latency_p50_ms', 0)} ms")
            st.write(f"latency p95: {op.get('latency_p95_ms', 0)} ms")
            cost = op.get("total_cost_usd")
            st.write("cost: n/a" if cost is None else f"cost: ${cost:.4f}")

    # --- per-run provenance + drill-in ------------------------------------------------
    st.subheader("Provenance & records")
    for r in selected:
        manifest = r["manifest"]
        with st.expander(_label(r)):
            git = manifest.get("git_sha")
            st.write(
                {
                    "tier": r["result"].get("tier"),
                    "git_sha": git[:10] if isinstance(git, str) else None,
                    "provider": manifest.get("provider"),
                    "model": manifest.get("model"),
                    "dataset_hash": manifest.get("dataset_hash"),
                }
            )
            for rec in r["result"].get("records", []):
                st.markdown(f"**Q. {rec.get('question', '')}**")
                st.write(rec.get("answer", ""))
                metrics = {**rec.get("retrieval", {}), **rec.get("answer_metrics", {})}
                if metrics:
                    st.caption(" · ".join(f"{k}={v:.3f}" for k, v in sorted(metrics.items())))


def main() -> None:
    render(load_runs(_runs_dir()))


if __name__ == "__main__":
    main()
