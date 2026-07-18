"""Run manifest — the reproducibility fingerprint written next to every run.

Determinism is a non-negotiable (spec §10): every run records enough to answer "what
produced these numbers?" — the config that was used, the git SHA of the code, the model
behind any judge, the exact dataset, and which evaluation tier ran. ``compare`` and the
regression gate (M5) rely on this to refuse apples-to-oranges comparisons later.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256_of_obj(obj: Any) -> str:
    """Stable hash of a JSON-able object (sorted keys → order-independent)."""
    encoded = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_of_file(path: Path) -> str | None:
    """Hash a file's bytes; None if it doesn't exist (golden set may be optional)."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str | None:
    """Best-effort ``git rev-parse HEAD``; None outside a repo so the harness stays
    portable (it must run fine from a plain ``pip install`` with no .git present)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None


@dataclass(slots=True)
class RunManifest:
    """Immutable record of the conditions under which a run was produced."""

    run_id: str
    created_at: str
    tier: str
    target_name: str
    config_hash: str | None = None
    dataset_hash: str | None = None
    git_sha: str | None = None
    provider: str | None = None
    model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        tier: str,
        target_name: str,
        config: dict[str, Any] | None = None,
        golden_path: Path | str | None = None,
        provider: str | None = None,
        model: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RunManifest:
        """Snapshot the current environment into a manifest."""
        golden = Path(golden_path) if golden_path is not None else None
        return cls(
            run_id=run_id,
            created_at=datetime.now(UTC).isoformat(),
            tier=tier,
            target_name=target_name,
            config_hash=_sha256_of_obj(config) if config is not None else None,
            dataset_hash=_sha256_of_file(golden) if golden is not None else None,
            git_sha=_git_sha(),
            provider=provider,
            model=model,
            extra=extra or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
