"""Golden-set loading. M1 ships the minimal JSONL reader the runner needs.

A golden record is deliberately small (spec §6): a ``question``, the ``relevant_doc_ids``
that *should* be retrieved, and a ``reference_answer``. The richer loaders/builder land in
M2/M6; this module owns only the on-disk shape so the manifest can hash it and the runner
can attach labels to each ``QueryRecord``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GoldenRecord:
    """One labelled evaluation example."""

    question: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    reference_answer: str | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> GoldenRecord:
        if "question" not in obj:
            raise ValueError(f"Golden record missing 'question': {obj!r}")
        return cls(
            question=str(obj["question"]),
            relevant_doc_ids=[str(x) for x in obj.get("relevant_doc_ids", [])],
            reference_answer=obj.get("reference_answer"),
        )


def load_golden(path: Path | str) -> list[GoldenRecord]:
    """Read a JSONL golden set. Blank lines are skipped; each line is one record."""
    p = Path(path)
    records: list[GoldenRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(GoldenRecord.from_dict(json.loads(line)))
    return records
