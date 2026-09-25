# Agent 1 - Task & Estimation Validation API

Validates Agile backlog tasks (title, description, scope, estimated hours)
using the Groq LLM, with Redis-backed result caching and pinned canonical
estimates so the same task always gets the same answer.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET  | `/health`, `/health/live`, `/health/ready` | `ready` also reports Redis status |
| POST | `/api/v1/task/validate` | single task (JSON), standard mode |
| POST | `/api/v1/task/validate-ai-assisted` | single task (JSON), AI-assisted mode |
| POST | `/api/v1/backlog/validate` | bulk tasks (JSON), standard mode |
| POST | `/api/v1/backlog/validate-ai-assisted` | bulk tasks (JSON), AI-assisted mode |
| POST | `/api/v1/backlog/upload-and-validate` | bulk tasks (`.xlsx` upload), standard mode |
| GET  | `/api/v1/cache/stats` | cache / pinned-estimate counts (standard + AI-assisted) |
| POST | `/api/v1/cache/clear` | force fresh validation for every task, both modes |

> The API has **no authentication** of its own. Keep it bound to
> localhost (see `docker-compose.yml`) and expose it only through the
> reverse proxy.

## AI-Assisted Estimation mode

The `-ai-assisted` endpoints accept the exact same request body as their
standard counterparts (`TaskInput` / `BulkTaskValidationRequest`) and
return the exact same response shape (`SingleTaskResponse` /
`BulkResponse`), so any UI already rendering an Agent 1 result can
render an AI-assisted result unchanged - only the button/endpoint the
UI calls differs.

What's different in AI-assisted mode:

- Effort is assessed assuming the task will be implemented with
  AI-assisted development tools. The model first independently derives
  an AI-assisted effort figure from the task's scope alone - without
  looking at the submitted `estimated_hours` - by reasoning about which
  parts of the described work AI tools can realistically accelerate and
  which parts remain human-dependent. Only after that independent
  derivation does it compare the result with the submitted estimate and
  apply the same decision rules (`PROCEED`, `REVIEW_ESTIMATE`,
  `REWRITE_TASK`, `REWRITE_AND_REESTIMATE`, `CANNOT_VALIDATE_ESTIMATE`).
- There is **no fixed AI productivity discount**. The same task can
  legitimately get a lower, similar, or identical AI-assisted estimate
  compared to standard mode, depending on how much of its real work is
  AI-accelerable.
- The insufficient-description rule (`CANNOT_VALIDATE_ESTIMATE`) is
  identical in both modes and always takes priority: AI assistance
  never compensates for a missing/vague description.
- Standard and AI-assisted results are cached and pinned completely
  separately (see Consistency (Redis) below) - validating a task in one
  mode never returns or contaminates the other mode's result, even for
  the exact same task content.

## Decisions

| Decision | `suggested_estimated_hours` |
|---|---|
| `PROCEED` | original estimate |
| `REWRITE_TASK` | original estimate |
| `REVIEW_ESTIMATE` | revised number |
| `REWRITE_AND_REESTIMATE` | revised number |
| `CANNOT_VALIDATE_ESTIMATE` | **`null`** - description insufficient; BA must update and resubmit |
| `ERROR` | original estimate (technical failure; counted in `failed_tasks`) |

`CANNOT_VALIDATE_ESTIMATE` is a valid business decision: it counts in
`validated_tasks`, never in `failed_tasks`, and is also reported in
`insufficient_description_tasks`. The rule is applied centrally, so single,
bulk JSON and Excel behave identically.

A description is rejected without an LLM call when it is empty or a
placeholder, shorter than `MIN_DESCRIPTION_WORDS`, or adds fewer than
`MIN_DESCRIPTION_NEW_WORDS` meaningful words beyond the title. Anything
else goes to the LLM, which may also return this decision.

## Excel columns

Header matching ignores case, spaces, `_` and `-`.

| Field | Accepted headers |
|---|---|
| task_id | Task ID |
| task_title | Task, Task Title, Task Name |
| task_description | Description, Task Description |
| estimated_hours | Estimated Hours, Estimate Hrs, Est Hrs, Planned Hours, Manual Estimate, Manual Hours, Estimate, Estimated Effort, Effort Hours |
| module_name / feature_name | Module, Feature |
| complexity | Complexity |
| project_tag | Project, Project Tag |
| assignee | Assignee, Assigned To, Employee Name |

Rows without a title are skipped. Rows without a Task ID get `ROW-<n>`.

## Consistency (Redis)

- Unchanged task -> identical cached result, no LLM call.
- Same task content, different estimate -> compared against the pinned
  canonical estimate, no LLM call.
- Changed title/description/scope -> fresh validation.
- Call `/api/v1/cache/clear` after changing the prompt.
- Standard mode uses Redis key prefixes `agent1:validation:` /
  `agent1:canonical:` (unchanged since before AI-assisted mode existed).
  AI-assisted mode uses separate prefixes `agent1ai:validation:` /
  `agent1ai:canonical:`, so the two modes can never share, overwrite, or
  return each other's cached/pinned result for the same task.

Without Redis the service still works, but the cache is in-memory and resets
on every restart.

## Local development (Windows)

```cmd
docker run -d --name agent1-redis -p 6379:6379 --restart unless-stopped redis:7
copy .env.example .env          & rem then set GROQ_API_KEY
pip install -r requirements.txt pytest httpx
uvicorn main:app --reload
```

## Tests

Groq is mocked, so no key or network is needed:

```cmd
set GROQ_API_KEY=dummy && pytest -q test_cannot_validate.py test_ai_assisted.py
```

`test_cannot_validate.py` covers the original standard-mode behavior.
`test_ai_assisted.py` covers the AI-assisted endpoints: independent
estimation, no fixed discount, insufficient-description handling, and
cache isolation between modes.

Run tests in a separate terminal from the server - the dummy key otherwise
overrides the real key in `.env`.
