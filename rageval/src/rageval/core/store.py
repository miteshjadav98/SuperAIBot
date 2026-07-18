"""Results store — plain-JSON persistence for runs. No database, on purpose.

Every run lands in ``runs/<run_id>/`` as two files: ``result.json`` (per-query records
+ operational rollups) and ``manifest.json`` (the reproducibility fingerprint). Reports,
``compare``, and the regression gate all read from here, so keeping it filesystem+JSON
means a fresh clone can inspect historical runs with zero services running.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rageval.core.manifest import RunManifest
from rageval.core.results import RunResult

RESULT_FILE = "result.json"
MANIFEST_FILE = "manifest.json"


@dataclass(slots=True)
class StoredRun:
    """A run read back from disk: the raw result + manifest dicts and their directory."""

    path: Path
    result: dict[str, Any]
    manifest: dict[str, Any]


class ResultsStore:
    """Reads and writes runs under a root directory (default ``runs/``)."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def save(self, result: RunResult, manifest: RunManifest) -> Path:
        """Persist a run; returns its directory. Deterministic key order for clean diffs."""
        target = self.run_dir(result.run_id)
        target.mkdir(parents=True, exist_ok=True)
        self._write_json(target / RESULT_FILE, result.to_dict())
        self._write_json(target / MANIFEST_FILE, manifest.to_dict())
        return target

    def save_result(self, result: RunResult) -> Path:
        """Re-persist only ``result.json`` (e.g. after scoring), leaving the manifest as-is.

        Metrics are computed *after* a run is produced, so they mustn't force a new
        manifest — the reproducibility fingerprint is fixed at run time, not scoring time.
        """
        target = self.run_dir(result.run_id)
        target.mkdir(parents=True, exist_ok=True)
        self._write_json(target / RESULT_FILE, result.to_dict())
        return target

    def load(self, run_id_or_path: str | Path) -> StoredRun:
        """Load a run by id (under root) or by explicit directory path."""
        path = Path(run_id_or_path)
        if not path.exists():
            path = self.run_dir(str(run_id_or_path))
        if not path.exists():
            raise FileNotFoundError(f"No run found at {run_id_or_path!r}")
        result = json.loads((path / RESULT_FILE).read_text(encoding="utf-8"))
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        return StoredRun(path=path, result=result, manifest=manifest)

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.write_text(
            json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
