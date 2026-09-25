# Agent 1 AI-Assisted Estimation — Production Change Summary

Baseline: existing production `main.py` from the supplied `agent1.zip`.

## Preserved
- Existing `/api/v1/task/validate` behavior and request/response contract.
- Existing `/api/v1/backlog/validate` behavior and request/response contract.
- Existing insufficient-description rule (`CANNOT_VALIDATE_ESTIMATE` + `suggested_estimated_hours: null`).
- Existing standard system prompt and standard cache/canonical fingerprints.
- Existing `/api/v1/cache/stats` response contract.
- Existing Groq, Redis, health, upload, and deployment behavior.

## Added
- `POST /api/v1/task/validate-ai-assisted`
- `POST /api/v1/backlog/validate-ai-assisted`
- AI-assisted system prompt that independently derives AI-assisted effort before comparing with submitted hours.
- Separate AI cache and canonical-estimate namespaces to prevent cross-mode contamination.
- Regression/AI-assisted tests.

## AI estimation rules
- Same title, description, scope, decision, and recommendation rules as standard mode.
- Insufficient descriptions are rejected before estimation.
- No fixed AI productivity percentage.
- AI-accelerable and human-dependent work are considered task-by-task.
- Submitted `estimated_hours` is compared only after an independent AI-assisted effort is derived.
