# Prompt Management Service

Versioned, **immutable** prompt storage on MongoDB Atlas with a FastAPI HTTP
API, audit logging, and cache-first reads (Redis or in-process).

Core idea: a prompt is a family of immutable version documents plus one
registry row that says which version is *active*. Updating a prompt always
appends version N+1; rolling back just moves the registry pointer. History is
never modified.

## Architecture

```text
prompt_service/
├── app/
│   ├── api/            # HTTP routes + dependency providers (thin layer)
│   │   ├── deps.py
│   │   └── routes.py
│   ├── services/       # business rules (versioning, rollback, caching)
│   │   └── prompt_service.py
│   ├── repositories/   # Mongo data access (repository pattern)
│   │   ├── prompt_repository.py
│   │   ├── registry_repository.py
│   │   └── audit_repository.py
│   ├── models/         # internal domain models
│   ├── schemas/        # Pydantic request/response contracts
│   ├── database/       # Motor client + index creation
│   ├── utils/          # cache backend, exceptions, logging
│   ├── config.py       # env-var settings (pydantic-settings)
│   └── main.py         # FastAPI app, lifespan, error mapping
├── tests/              # unit tests with in-memory fakes (no DB needed)
├── requirements.txt
└── .env.example
```

Layering: **routes → service → repositories → Mongo**. The service never
imports FastAPI; the routes never touch Mongo. Each layer depends on the one
below through constructor injection, so every piece is swappable and testable
in isolation (SOLID: single responsibility + dependency inversion).

## Data model

`prompts` — one document per version, immutable:

```json
{
  "prompt_id": "rag_system_prompt",
  "version": 3,
  "name": "RAG System Prompt",
  "content": "You are a retrieval-augmented assistant...",
  "description": "Tightened citation rules",
  "created_by": "admin",
  "created_at": "2026-07-11T10:00:00Z",
  "metadata": { "model": "gpt-4.1", "temperature": 0.2 }
}
```

`prompt_registry` — one row per prompt, the single source of truth for
what's active:

```json
{
  "prompt_id": "rag_system_prompt",
  "current_version": 3,
  "latest_version": 4,
  "updated_at": "2026-07-11T10:05:00Z"
}
```

`prompt_audit` — append-only log of every CREATE_PROMPT / CREATE_VERSION /
ROLLBACK with who and when.

Design decisions vs. the original sketch:

- **No `status` field on versions.** With a registry pointer, a second
  "active" signal can only disagree with it. The registry alone decides.
- **`latest_version` lives in the registry** in addition to
  `current_version`. After a rollback (current=2, latest=4), a new version
  becomes 5 — computed without scanning the prompts collection and without
  colliding with history.
- **Concurrency safety** comes from the unique `(prompt_id, version)` index:
  two writers racing for the same number can't both insert; the loser retries
  with the next number (optimistic concurrency, max 3 attempts → 409).

Indexes (created automatically at startup):

```python
db.prompts.create_index([("prompt_id", 1), ("version", -1)], unique=True)
db.prompt_registry.create_index([("prompt_id", 1)], unique=True)
db.prompt_audit.create_index([("prompt_id", 1), ("timestamp", -1)])
```

## Caching

`GET /prompts/{prompt_id}` is cache-first: hit → served from cache; miss →
Mongo, then cached for `CACHE_TTL_SECONDS`. Publishing a new version and
rolling back both **invalidate** the key (`prompt:active:{prompt_id}`).

- `REDIS_URL` set → Redis (`redis.asyncio`), correct across multiple
  service instances.
- `REDIS_URL` unset → in-process TTL cache, zero infra for a single instance.

Cache failures degrade gracefully: a Redis outage turns reads into DB reads,
never into errors.

## Running

```powershell
cd prompt_service
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then fill in MONGODB_URI
.venv\Scripts\uvicorn app.main:app --reload --port 8020
```

Swagger UI: http://localhost:8020/docs

## Hosting on the Azure VM (Swagger over the public URL)

`deployment_script.sh` deploys this service alongside the rest of the
platform: systemd unit `aibot-prompts` runs uvicorn on `127.0.0.1:8020`, and
nginx exposes it under the stripped `/prompt-api/` prefix. The unit sets
`ROOT_PATH=/prompt-api`, which is what makes the hosted Swagger UI work —
without it, `/docs` would try to load `openapi.json` and send "Try it out"
requests to the wrong (unprefixed) paths.

After deploying, open and test the API in the browser at:

**https://aibot.miteklabs.tech/prompt-api/docs**

(Endpoints are prefixed on the VM, e.g.
`https://aibot.miteklabs.tech/prompt-api/prompts/rag_system_prompt`.)

### Protecting the public API

Anyone who can reach the URL can otherwise edit your prompts. Add
`PROMPT_API_KEY=<secret>` to the shared `.env` and restart `aibot-prompts`;
every `/prompts` endpoint then requires the `X-API-Key` header (401 without
it). In Swagger UI, click **Authorize**, paste the key once, and all
"Try it out" calls send it automatically. When the variable is unset (local
dev), auth is disabled.

Tests (no database required — the service layer is tested against in-memory
fakes that reproduce Mongo's unique-index behaviour):

```powershell
.venv\Scripts\pytest -q
```

## API reference with examples

### POST /prompts — create a prompt (version 1)

Request:

```json
{
  "prompt_id": "rag_system_prompt",
  "name": "RAG System Prompt",
  "content": "You are a retrieval-augmented assistant. Cite sources.",
  "description": "Initial version",
  "created_by": "admin",
  "metadata": { "model": "gpt-4.1", "temperature": 0.2 }
}
```

Response `201`:

```json
{
  "prompt_id": "rag_system_prompt",
  "version": 1,
  "name": "RAG System Prompt",
  "content": "You are a retrieval-augmented assistant. Cite sources.",
  "description": "Initial version",
  "created_by": "admin",
  "created_at": "2026-07-11T09:00:00Z",
  "metadata": { "model": "gpt-4.1", "temperature": 0.2 },
  "is_active": true
}
```

Errors: `409` if `prompt_id` already exists, `422` on invalid body.

### PUT /prompts/{prompt_id} — publish a new version

Never overwrites. `name`/`metadata` are optional and inherited from the
latest version when omitted.

Request:

```json
{
  "content": "You are a retrieval-augmented assistant. Cite sources inline as [n].",
  "description": "Tightened citation rules",
  "created_by": "admin"
}
```

Response `201`:

```json
{
  "prompt_id": "rag_system_prompt",
  "version": 2,
  "name": "RAG System Prompt",
  "content": "You are a retrieval-augmented assistant. Cite sources inline as [n].",
  "description": "Tightened citation rules",
  "created_by": "admin",
  "created_at": "2026-07-11T10:00:00Z",
  "metadata": { "model": "gpt-4.1", "temperature": 0.2 },
  "is_active": true
}
```

Errors: `404` unknown prompt, `409` if concurrent writers exhaust retries.

### GET /prompts/{prompt_id} — active version (cache-first)

Response `200`: same shape as above, always the version the registry points
at. Errors: `404`.

### GET /prompts/{prompt_id}/versions — full history

Response `200`:

```json
{
  "prompt_id": "rag_system_prompt",
  "current_version": 2,
  "total": 2,
  "versions": [
    { "version": 1, "is_active": false, "content": "...", "...": "..." },
    { "version": 2, "is_active": true,  "content": "...", "...": "..." }
  ]
}
```

### GET /prompts/{prompt_id}/versions/{version} — one version

Response `200`: single version document (with `is_active`). Errors: `404`
for unknown prompt or version.

### POST /prompts/{prompt_id}/rollback/{version} — roll back

Optional body: `{ "performed_by": "admin" }`

Response `200`:

```json
{
  "prompt_id": "rag_system_prompt",
  "rolled_back_to": 1,
  "previous_version": 2,
  "updated_at": "2026-07-11T11:00:00Z"
}
```

Only the registry pointer moves; the audit log records
`ROLLBACK v1 (from v2)`. A later PUT creates version 3 (latest+1), so
history never collides.

### GET /prompts — list all prompts

Response `200`:

```json
[
  {
    "prompt_id": "rag_system_prompt",
    "current_version": 1,
    "latest_version": 2,
    "updated_at": "2026-07-11T11:00:00Z"
  }
]
```

## Consuming prompts from an application

```python
import httpx

async def get_prompt(prompt_id: str) -> str:
    async with httpx.AsyncClient(base_url="http://localhost:8010") as client:
        resp = await client.get(f"/prompts/{prompt_id}")
        resp.raise_for_status()
        return resp.json()["content"]
```

Because reads are cache-first with invalidation on publish, applications can
fetch the active prompt on every request without hammering Mongo — and a
rollback takes effect immediately.
