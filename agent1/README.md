# Agent 1 - Task & Estimation Validation API

Validates Agile backlog tasks (title, description, scope, estimated hours)
using the Groq LLM. Pure JSON API - no file upload/parsing.

## Architecture

```
Client
  │
  ├─ GET  /health/live, /health/ready        (no auth)
  │
  └─ X-API-Key required ──────────────────────────────────
       │
       ├─ POST /api/v1/task/validate          synchronous, single task
       │         └─ validate_with_groq()
       │               ├─ TaskEstimationStore.get_history(task_id) ── Redis
       │               ├─ GroqRateGate ─────────────────────────── Groq API
       │               ├─ is_runaway_revision()  (code-level, no LLM call)
       │               └─ TaskEstimationStore.record_round(task_id) ── Redis
       │
       ├─ POST /api/v1/backlog/validate       returns 202 immediately
       │         └─ enqueue_job() → JobStore.create()
       │                          → background thread (daemon)
       │                              └─ submits each task to a shared
       │                                 ThreadPoolExecutor(BULK_MAX_WORKERS)
       │                              └─ each task independently goes
       │                                 through validate_with_groq()
       │                                 (same Redis-backed history lookup
       │                                 and drift guard as the sync path)
       │                              └─ JobStore.record_result() per task
       │                              └─ JobStore.finalize()
       │
       └─ GET  /api/v1/backlog/jobs/{job_id}  poll job progress/results
                 └─ JobStore.get()
```

Key building blocks:

- **`GroqRateGate`** - a `threading.Semaphore` (concurrency) plus a
  sliding-window `threading.Condition` (requests/minute) wrapped in one
  context manager. Every Groq call goes through it, regardless of which
  endpoint or thread issued it, so the two limits are enforced globally.
- **`LlmCallBudget`** - a per-task counter shared across the initial
  validation attempt loop, one decision-repair round-trip, and one
  estimate-correction round-trip. Once `MAX_LLM_CALLS_PER_TASK` is spent,
  the task fails fast with `LLM_CALL_BUDGET_EXCEEDED` instead of chaining
  further retries.
- **`JobStore` (ABC) / `InMemoryJobStore`** - thread-safe, single lock,
  in-memory dict of `JobRecord`s. Swappable for a Postgres/Redis-backed
  store later without touching endpoint code (it's injected via
  `Depends(get_job_store)`).
- **Background jobs run on plain `threading.Thread`s** (not asyncio), one
  coordinator thread per job, which itself fans out per-task work onto a
  shared `ThreadPoolExecutor` sized by `BULK_MAX_WORKERS`. This keeps
  total concurrent Groq-bound threads bounded across all jobs at once,
  not just within one job.
- **`TaskEstimationStore` (ABC) / `RedisTaskEstimationStore`** - the same
  isolation pattern as `JobStore`, but backed by Redis instead of memory
  from the start, since this history specifically needs to survive
  restarts and be shared across replicas (see
  [Repeated verification / estimate drift protection](#repeated-verification--estimate-drift-protection)
  below). Injected via `Depends(get_task_estimation_store)`. A
  synchronous `redis.Redis` client is used deliberately, not
  `redis.asyncio` - it matches this file's existing threading-based
  concurrency model (background job threads and `ThreadPoolExecutor`
  workers have no asyncio event loop to await into), and every call sets
  short socket timeouts so a down Redis can't block a request for long.

## Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| GET | `/health/live` | none | process liveness only |
| GET | `/health/ready` | none | config completeness only, never calls Groq |
| POST | `/api/v1/task/validate` | `X-API-Key` | synchronous, one task |
| POST | `/api/v1/backlog/validate` | `X-API-Key` | returns 202 + `job_id` immediately |
| GET | `/api/v1/backlog/jobs/{job_id}` | `X-API-Key` | poll status/results |

Swagger UI: `/docs` · ReDoc: `/redoc`

## Repeated verification / estimate drift protection

**The problem this solves:** a task creator manually estimates a task
(say 4 hours) and calls `POST /api/v1/task/validate`. Agent 1 suggests
12. The creator copies that 12 into their tool's "estimated hours" field,
overwriting their own original number, and calls `/api/v1/task/validate`
again with the *same* `task_id` and `estimated_hours=12`. Without any
memory of the first call, Agent 1 judges the task fresh and may suggest
an even higher number - estimates can drift upward (or downward)
indefinitely across repeated verify-edit-verify cycles.

**How it's solved:** the server itself remembers, keyed by `task_id`,
persisted in Redis so the memory survives restarts and is visible to
every replica. Clients don't send anything extra - `task_id` already
exists on every request today, so this is transparent from the first
repeat call. Two layers:

1. **Prompt-level nudge.** Before calling Groq, prior rounds for this
   `task_id` are looked up and included in the prompt (inside the usual
   `<task_data>` untrusted-content tags - history is data, not
   instructions, same as everything else). The model is told this is a
   revision, to compare the current estimate against the *original*
   first-ever estimate on file, and to lean toward holding the estimate
   rather than revising it again if the task hasn't materially changed
   and the creator simply adopted the AI's own prior suggestion.
2. **Code-level hard guard (`is_runaway_revision`).** The model's own
   judgment isn't trusted alone. After the model responds, a plain
   Python check looks at the estimated/suggested values across the
   task's rounds (including the current proposal): if the last three
   consecutive values are moving strictly in the same direction (all
   increasing or all decreasing), the result is force-overridden to
   `CANNOT_VALIDATE_ESTIMATE`, `suggested_estimated_hours` is reset back
   to the currently-submitted value (no further change proposed), and
   `recommendation` explains that the task needs human re-scoping. This
   check costs no extra Groq call.

Every finalized round (whichever decision it ends up with) is recorded
back to Redis afterward - **except** `ERROR` results from Groq failures,
which aren't real estimation rounds and would otherwise poison future
drift checks.

**Fail open:** if Redis is unreachable - at startup, or on any individual
call - every `TaskEstimationStore` operation logs a warning and behaves
as "no history available." Validation proceeds exactly as it did before
this feature existed; the task is judged fresh, with no drift
protection, rather than the request failing. `/health/ready` surfaces a
`redis_reachable` flag for ops visibility, but Redis being down does
**not** make the service report `NOT_READY` - see the code comment on
`health_ready()` for the reasoning (drift protection is a safety
enhancement on top of validation, not core functionality).

### Example: guard triggering on a repeat verification

Call 1 - initial manual estimate:

```json
// POST /api/v1/task/validate  { "task_id": "TASK-101", "estimated_hours": 4, ... }
{
  "status": "COMPLETED",
  "result": {
    "task_id": "TASK-101",
    "decision": "REVIEW_ESTIMATE",
    "suggested_estimated_hours": 12,
    "recommendation": "4 hours appears low for this scope; 12 hours is a more realistic estimate. This is an AI-generated estimation review based only on the supplied task information; human approval is required.",
    "...": "..."
  }
}
```

Call 2 - the creator copies `12` into `estimated_hours` and re-submits
the *same* `task_id`. If the model (still) wants to push the number
further in the same direction, the hard guard intervenes:

```json
// POST /api/v1/task/validate  { "task_id": "TASK-101", "estimated_hours": 12, ... }
{
  "status": "COMPLETED",
  "result": {
    "task_id": "TASK-101",
    "decision": "CANNOT_VALIDATE_ESTIMATE",
    "suggested_estimated_hours": 12,
    "recommendation": "This task's estimate has moved in the same direction across multiple verification rounds without a material scope change. This pattern indicates the task needs human re-scoping rather than another automated estimate. Human review is required. This is an AI-generated estimation review based only on the supplied task information; human approval is required.",
    "...": "..."
  }
}
```

`suggested_estimated_hours` holds at the submitted value (`12`) rather
than proposing a third, higher number - the loop is broken and a human
is asked to re-scope the task instead.

## Sample requests

### Single task

```bash
curl -X POST http://localhost:8001/api/v1/task/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AGENT_API_KEY" \
  -d '{
    "task_id": "TASK-101",
    "module_name": "Billing",
    "feature_name": "Invoicing",
    "task_title": "Add PDF export to invoice list",
    "task_description": "Add a button on the invoice list page that exports the currently filtered invoices to a single PDF file, matching the existing print layout.",
    "estimated_hours": 8,
    "complexity": "Medium",
    "project_tag": "ERP-CORE",
    "assignee": "jane.doe",
    "task_type": "Feature",
    "reuse_expected": true,
    "reuse_percentage": 40,
    "existing_components": ["PdfExportService", "InvoiceListFilters"],
    "dependencies": ["TASK-090"],
    "acceptance_criteria": [
      "Export button is visible only when at least one invoice is listed",
      "Exported PDF matches the current filter set"
    ],
    "historical_similar_tasks": [
      {"task_title": "Add PDF export to order list", "actual_hours": 10, "complexity": "Medium", "similarity_notes": "Same export service, different list"}
    ]
  }'
```

Response (`200`):

```json
{
  "status": "COMPLETED",
  "result": {
    "task_id": "TASK-101",
    "decision": "PROCEED",
    "task_title_assessment": "Clear and action-oriented.",
    "task_description_assessment": "Scope and boundary are well defined.",
    "scope_assessment": "Scope is appropriately bounded to the invoice list page.",
    "effort_assessment": "8 hours appears reasonable given 40% reuse of PdfExportService and a similar historical task at 10 hours.",
    "suggested_task_title": "Add PDF export to invoice list",
    "suggested_task_description": "Add a button on the invoice list page that exports the currently filtered invoices to a single PDF file, matching the existing print layout.",
    "suggested_estimated_hours": 8,
    "confidence_score": 0.72,
    "recommendation": "Estimate appears reasonable; proceed as planned. This is an AI-generated estimation review based only on the supplied task information; human approval is required."
  }
}
```

### Bulk validation (async job)

```bash
curl -i -X POST http://localhost:8001/api/v1/backlog/validate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AGENT_API_KEY" \
  -d '{
    "project": "ERP-CORE",
    "sprint": "Sprint 24",
    "tasks": [
      {"task_id": "TASK-101", "task_title": "Add PDF export to invoice list", "estimated_hours": 8},
      {"task_id": "TASK-102", "task_title": "Fix login timeout", "estimated_hours": 2}
    ]
  }'
```

Response (`202`):

```json
{
  "job_id": "250fbb7c647a413eb5e9fbf544ce4e10",
  "status": "QUEUED",
  "total_tasks": 2,
  "completed_tasks": 0,
  "failed_tasks": 0,
  "results": []
}
```

Poll for progress/results:

```bash
curl http://localhost:8001/api/v1/backlog/jobs/250fbb7c647a413eb5e9fbf544ce4e10 \
  -H "X-API-Key: $AGENT_API_KEY"
```

While running: `status` is `PROCESSING`, `results` stays `[]`.
Once finished: `status` is `COMPLETED` / `PARTIALLY_COMPLETED` / `FAILED`,
and `results` contains one `ValidationResult` per task, in original
input order.

### Health

```bash
curl http://localhost:8001/health/live
curl http://localhost:8001/health/ready
```

`/health/ready` returns `503` with `{"status":"NOT_READY","checks":{...}}`
if `GROQ_API_KEY`, `GROQ_MODEL`, or `AGENT_API_KEY` isn't configured, or
the service is mid-shutdown. It never calls Groq.

### Errors

Every non-2xx response (except FastAPI's own request-body 422s, which
still use the same envelope) has the shape:

```json
{"error": {"code": "UNAUTHORIZED", "message": "Missing or invalid API key."}}
```

Public error codes in use: `UNAUTHORIZED`, `JOB_NOT_FOUND`,
`VALIDATION_ERROR`, `SERVICE_SHUTTING_DOWN`, `LLM_AUTHENTICATION_FAILED`,
`INTERNAL_VALIDATION_ERROR`. Per-task Groq failures (timeouts, rate
limits, invalid model output, etc.) do **not** raise HTTP errors - they
degrade to a `ValidationResult` with `"decision": "ERROR"` and a
`recommendation` naming the code (`LLM_TIMEOUT`, `LLM_RATE_LIMITED`,
`LLM_TRANSIENT_ERROR`, `LLM_PROVIDER_ERROR`, `LLM_CALL_BUDGET_EXCEEDED`,
`INVALID_MODEL_DECISION`, `ESTIMATE_CORRECTION_FAILED`,
`INTERNAL_VALIDATION_ERROR`), so one bad task never fails a whole batch.
The one exception is `LLM_AUTHENTICATION_FAILED`: since a bad Groq key
fails identically for every task, it still short-circuits the rest of
the batch (queued-but-not-started tasks are cancelled rather than each
independently retried against a dead key).

## Configuration

See [`.env.example`](.env.example) for the full list with defaults. All
tuning variables are optional; only `GROQ_API_KEY`, `GROQ_MODEL`, and
`AGENT_API_KEY` need to be set for a working deployment - the service
starts without them, but `/health/ready` reports `NOT_READY` and
protected endpoints reject requests until they're set.

`REDIS_URL` (default `redis://localhost:6379/0`, overridden to
`redis://redis:6379/0` under docker-compose) and
`TASK_ESTIMATION_RETENTION_SECONDS` (default `86400`) back estimate-drift
protection specifically - see
[Repeated verification / estimate drift protection](#repeated-verification--estimate-drift-protection).
They're optional in the same fail-open sense: core task validation works
with no Redis at all, just without drift protection.

## Summary of improvements over the previous version

1. Removed the Excel upload path entirely (`pandas`, `openpyxl`,
   `python-multipart`, `UploadFile` - all gone). Pure JSON API.
2. Bulk validation is now job-based: `POST /api/v1/backlog/validate`
   returns `202` immediately; a background thread processes the batch
   and `GET /api/v1/backlog/jobs/{job_id}` reports progress/results.
   Fixes proxy/nginx/Cloudflare/client timeouts on large batches.
3. `X-API-Key` auth (`AGENT_API_KEY`) on every endpoint except
   `/health/*`, checked with `secrets.compare_digest`.
4. Split health into `/health/live` (process up) and `/health/ready`
   (config completeness; never calls Groq; `503` if incomplete). Startup
   no longer crashes on a missing `GROQ_API_KEY`.
5. Added `GROQ_MAX_REQUESTS_PER_MINUTE` as a true sliding-window RPM
   limiter (`GroqRateGate`), alongside the existing concurrency limit -
   thread-safe, blocking, no busy-waiting, logs when a call has to wait.
6. Added `MAX_LLM_CALLS_PER_TASK` as a hard ceiling shared across the
   initial attempt loop, decision repair, and estimate correction, so one
   task can no longer chain an unbounded number of Groq calls.
7. Extended `TaskInput` with optional estimation context (`task_type`,
   `reuse_expected`, `reuse_percentage`, `existing_components`,
   `dependencies`, `acceptance_criteria`, `historical_similar_tasks`) and
   updated the prompt to use it with cautious wording, never inventing
   missing information.
8. All task content is wrapped in `<task_data>` tags with an explicit
   system-prompt rule that it is data, never instructions, addressing
   prompt-injection risk.
9. Unsupported `decision` values now trigger one repair attempt before
   falling back to `ERROR` / `INVALID_MODEL_DECISION` - no more silent
   acceptance of arbitrary model output.
10. Structured, non-leaking error responses everywhere
    (`{"error": {"code", "message"}}`); stack traces, provider payloads,
    and env vars never reach the client, only server logs.
11. Request-level validation: empty/duplicate `task_id`, empty
    `task_title`, oversized titles/descriptions, too many
    historical-task entries, and too many tasks per job are all rejected
    with `422` (`MAX_TITLE_CHARS`, `MAX_DESCRIPTION_CHARS`,
    `MAX_HISTORICAL_TASKS`, `MAX_TASKS_PER_JOB`).
12. Structured single-line logs with `request_id`/`job_id`/`task_id`,
    elapsed time, Groq call count, and final decision on every
    validation.
13. Graceful shutdown: new jobs are rejected once shutdown begins,
    in-flight job threads get a bounded window to finish, then the
    shared executor is closed cleanly.
14. Removed all pandas-only code (`hours_to_decimal`, column aliasing,
    `dataframe_to_tasks`) now that input is always typed JSON.

## Remaining limitations

- The **job store** (`JobStore` / bulk validation progress) is still
  in-memory and per-process: jobs are lost on restart and don't survive
  horizontal scaling (multiple replicas each have their own job set).
  It's isolated behind the `JobStore` interface specifically so it can be
  swapped for a persistent backend later. This no longer applies to
  estimate-drift history, which is Redis-backed and does survive
  restarts and horizontal scaling - see the next point.
- **Estimate-drift history** depends on Redis being reachable to actually
  provide protection; it's a genuine new runtime dependency (see
  `docker-compose.yml`'s `redis` service). Every read/write fails open on
  a Redis outage - task validation itself is unaffected, but drift
  protection is silently disabled until Redis is reachable again, and
  each failed call adds up to `~2 × socket timeout` (a few seconds) of
  latency to that task's validation while it retries against a down
  Redis. There's no separate circuit breaker to short-circuit repeated
  failures faster than that within a single outage.
- No request body size limit is enforced in Python beyond the per-field
  character caps; put a reverse proxy in front (nginx
  `client_max_body_size`, etc.) for defense-in-depth against very large
  payloads.
- Background job threads are daemon threads coordinating work on a
  shared executor; there's no persisted work queue, so an ungraceful
  process kill (not a clean shutdown) drops any jobs that were still
  `PROCESSING`.
- Rate limiting and the call budget are process-local; running multiple
  replicas means each enforces its own `GROQ_MAX_REQUESTS_PER_MINUTE`
  independently (the effective account-wide RPM is `N × limit`).
- `MAX_LLM_RETRIES` and `MAX_LLM_CALLS_PER_TASK` interact but aren't
  auto-reconciled: setting `MAX_LLM_CALLS_PER_TASK` lower than
  `MAX_LLM_RETRIES` will cut the very first validation attempt loop
  short. The default formula (`max(6, MAX_LLM_RETRIES * 2)`) keeps this
  safe unless both are overridden inconsistently in `.env`.
- `is_runaway_revision`'s monotonic-direction check is a heuristic: a
  task that genuinely grows in scope across two real revisions (not a
  copy-the-suggestion-back-in loop) can also trip it, forcing
  `CANNOT_VALIDATE_ESTIMATE` and asking for human re-scoping. This is
  treated as the correct trade-off (ambiguous compounding change should
  go to a human either way), not a bug, but it's worth knowing the guard
  can't distinguish "drift" from "legitimately growing scope" - only the
  human reviewer can.
