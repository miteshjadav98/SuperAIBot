"""Central prompt store for agents.

Agents load their prompts from MongoDB — the same ``prompts`` /
``prompt_registry`` collections the standalone Prompt Management service
(``prompt_service/``) manages — instead of hardcoding them. Every prompt is
therefore versioned and can be edited or rolled back from the Prompt
Management API without changing code.

Behaviour:
- **Cache-first** with a short TTL so repeated loads are cheap.
- **Auto-seed**: the first time a prompt is requested it is inserted as
  version 1 from the code-provided default, so new prompts show up in the
  versioning UI automatically (no manual bootstrap step).
- **Fail-safe**: if Mongo is unset or unreachable, the code default is
  returned, so an agent never breaks because the prompt store is down.

Prompts are read when an agent graph is built (import time). After editing a
prompt via the Prompt Management API, restart the agent service — or call
``clear_cache()`` — so the process picks up the new active version.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)

# Long enough that repeated reads within one request are free, short enough
# that a rebuilt/redeployed agent sees edits quickly.
_CACHE_TTL_SECONDS = 60.0

_cache: dict[str, tuple[float, str]] = {}
_lock = Lock()

# Every prompt id the loaded code owns, with the default it ships.
# Populated by `get_prompt` and `Prompt`, read by `scripts/publish_prompts.py`
# — which exists because seeding is one-way: once a prompt is in Mongo, editing
# the code default changes nothing that is already deployed.
_declared: dict[str, tuple[str, str]] = {}  # prompt_id -> (name, default content)


def register_default(prompt_id: str, default: str, *, name: str | None = None) -> str:
    """Declare a prompt's code default. Returns ``default`` unchanged."""
    _declared[prompt_id] = (name or prompt_id, default)
    return default


def declared_defaults() -> dict[str, tuple[str, str]]:
    """Every prompt the imported modules own. Import the agents first — a prompt
    is declared when its module is imported, not before."""
    return dict(_declared)


class Prompt:
    """A prompt whose default is declared at import but loaded at call time.

    Agents call :func:`get_prompt` while building, so importing them is enough
    to declare their prompts. Prompts loaded *inside* a function (the Super
    Bot's, which re-render per turn) would otherwise stay invisible to the
    publisher until someone happened to run a turn.
    """

    def __init__(self, prompt_id: str, default: str, *, name: str | None = None) -> None:
        self.id = prompt_id
        self.name = name or prompt_id
        self.default = register_default(prompt_id, default, name=name)

    def get(self) -> str:
        return get_prompt(self.id, self.default, name=self.name)


def get_prompt(
    prompt_id: str,
    default: str,
    *,
    name: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Return the active content for ``prompt_id``.

    Reads the version pointed at by ``prompt_registry.current_version``. If the
    prompt does not exist yet, it is seeded as version 1 from ``default``. On
    any error the ``default`` is returned so callers never fail.

    ``name``/``metadata`` are only used when seeding (they describe the prompt
    in the versioning UI).
    """
    register_default(prompt_id, default, name=name)

    now = time.monotonic()
    cached = _cache.get(prompt_id)
    if cached is not None and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    content = _resolve(prompt_id, default, name, metadata)
    with _lock:
        _cache[prompt_id] = (now, content)
    return content


def _resolve(
    prompt_id: str, default: str, name: str | None, metadata: dict | None
) -> str:
    from core import db

    if not db.mongo_configured():
        return default
    try:
        prompts = db.get_db()["prompts"]
        registry = db.get_db()["prompt_registry"]

        entry = registry.find_one({"prompt_id": prompt_id})
        if entry:
            doc = prompts.find_one(
                {"prompt_id": prompt_id, "version": entry.get("current_version", 1)}
            )
            if doc and doc.get("content"):
                return doc["content"]

        _seed(prompts, registry, prompt_id, default, name, metadata)
    except Exception as exc:  # noqa: BLE001 — never let prompt loading break an agent
        logger.warning(
            "prompt '%s' load failed (%s); using code default", prompt_id, exc
        )
    return default


def _seed(
    prompts,
    registry,
    prompt_id: str,
    default: str,
    name: str | None,
    metadata: dict | None,
) -> None:
    """Insert version 1 from the code default. Idempotent via upsert, so two
    workers seeding the same prompt concurrently is harmless."""
    now = datetime.now(timezone.utc)
    prompts.update_one(
        {"prompt_id": prompt_id, "version": 1},
        {
            "$setOnInsert": {
                "prompt_id": prompt_id,
                "version": 1,
                "name": name or prompt_id,
                "content": default,
                "description": "Auto-seeded from code default",
                "created_by": "system",
                "created_at": now,
                "metadata": metadata or {},
            }
        },
        upsert=True,
    )
    registry.update_one(
        {"prompt_id": prompt_id},
        {
            "$setOnInsert": {
                "prompt_id": prompt_id,
                "current_version": 1,
                "latest_version": 1,
                "updated_at": now,
            }
        },
        upsert=True,
    )
    logger.info("Seeded prompt '%s' v1 from code default", prompt_id)


# --- Publishing --------------------------------------------------------------


def active_content(prompt_id: str) -> str | None:
    """The stored active content for ``prompt_id``, or ``None`` if unseeded.

    Unlike :func:`get_prompt` this never falls back to a code default and never
    seeds — the publisher needs to know what is *actually* in the database.
    """
    from core import db

    if not db.mongo_configured():
        return None

    entry = db.get_db()["prompt_registry"].find_one({"prompt_id": prompt_id})
    if not entry:
        return None
    doc = db.get_db()["prompts"].find_one(
        {"prompt_id": prompt_id, "version": entry.get("current_version", 1)}
    )
    return doc.get("content") if doc else None


def publish(
    prompt_id: str,
    content: str,
    *,
    name: str | None = None,
    description: str = "Published from code default",
    performed_by: str = "system",
) -> int:
    """Store ``content`` as the next version and make it active.

    Additive, exactly like the Prompt Management service: existing versions are
    immutable, so the previous one stays one rollback away. Returns the new
    version number.
    """
    from core import db

    if not db.mongo_configured():
        raise RuntimeError("MONGODB_URI is not set, so there is nowhere to publish.")

    prompts = db.get_db()["prompts"]
    registry = db.get_db()["prompt_registry"]
    now = datetime.now(timezone.utc)

    entry = registry.find_one({"prompt_id": prompt_id}) or {}
    version = int(entry.get("latest_version", 0)) + 1

    prompts.insert_one(
        {
            "prompt_id": prompt_id,
            "version": version,
            "name": name or prompt_id,
            "content": content,
            "description": description,
            "created_by": performed_by,
            "created_at": now,
            "metadata": {},
        }
    )
    registry.update_one(
        {"prompt_id": prompt_id},
        {"$set": {"current_version": version, "latest_version": version, "updated_at": now}},
        upsert=True,
    )
    # Best-effort audit, same rule the prompt service follows: an audit failure
    # must not undo a completed publish.
    try:
        db.get_db()["prompt_audit"].insert_one(
            {
                "prompt_id": prompt_id,
                "version": version,
                "action": "CREATE_VERSION" if version > 1 else "CREATE_PROMPT",
                "performed_by": performed_by,
                "timestamp": now,
                "detail": description,
            }
        )
    except Exception:  # noqa: BLE001
        logger.exception("audit write failed for %s v%s", prompt_id, version)

    clear_cache()
    return version


def clear_cache() -> None:
    """Drop cached prompts so the next ``get_prompt`` re-reads Mongo."""
    with _lock:
        _cache.clear()
