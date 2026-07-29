from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)
from pydantic import BaseModel, Field


load_dotenv()

# ==============================================================
# CONFIGURATION
# ==============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()

# How many backlog tasks are validated in parallel during a bulk request.
# Kept low by default because Groq's free tier has fairly tight per-minute
# request/token limits; each unit here is one full chat-completion call.
# Override with BULK_MAX_WORKERS in .env if your tier allows more headroom.
BULK_MAX_WORKERS = max(
    1,
    int(os.getenv("BULK_MAX_WORKERS", "3")),
)

# Client-side ceiling on concurrent Groq calls, independent of thread pool
# size. This is the actual backpressure valve: even if BULK_MAX_WORKERS is
# raised, no more than this many requests are ever in flight against Groq
# at once, which is what keeps a burst upload from tripping the account's
# RPM (requests-per-minute) limit.
GROQ_MAX_CONCURRENT_REQUESTS = max(
    1,
    int(os.getenv("GROQ_MAX_CONCURRENT_REQUESTS", "3")),
)

GROQ_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "60")
)
MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "3"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))

# Long descriptions cost tokens without adding validation value beyond a
# point; truncating keeps a single task from eating a disproportionate
# share of the per-minute token budget.
MAX_DESCRIPTION_CHARS = int(os.getenv("MAX_DESCRIPTION_CHARS", "4000"))

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent1")

client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_REQUEST_TIMEOUT_SECONDS)

# Shared across all worker threads in a bulk request so total in-flight
# Groq calls never exceed GROQ_MAX_CONCURRENT_REQUESTS, regardless of
# BULK_MAX_WORKERS.
_groq_concurrency_gate = threading.Semaphore(GROQ_MAX_CONCURRENT_REQUESTS)

app = FastAPI(
    title="Boscosoft Task Validation API",
    version="1.3.0",
)


class TaskInput(BaseModel):
    task_id: str
    module_name: str = ""
    feature_name: str = ""
    task_title: str
    task_description: str = ""
    estimated_hours: float = Field(ge=0)
    complexity: str = ""
    project_tag: str = ""
    assignee: str = ""


Decision = Literal[
    "PROCEED",
    "REVIEW_ESTIMATE",
    "REWRITE_TASK",
    "REWRITE_AND_REESTIMATE",
    "CANNOT_VALIDATE_ESTIMATE",
    "ERROR",
]


class ValidationResult(BaseModel):
    task_id: str
    decision: Decision

    task_title_assessment: str
    task_description_assessment: str
    scope_assessment: str
    effort_assessment: str

    # These fields always contain usable values.
    # They contain either the original value or the corrected value.
    suggested_task_title: str
    suggested_task_description: str
    suggested_estimated_hours: float

    confidence_score: float = Field(ge=0, le=1)
    recommendation: str


class SingleTaskResponse(BaseModel):
    status: Literal["COMPLETED", "FAILED"]
    result: ValidationResult


class BulkResponse(BaseModel):
    status: Literal[
        "COMPLETED",
        "PARTIALLY_COMPLETED",
        "FAILED",
    ]
    total_tasks: int
    validated_tasks: int
    failed_tasks: int
    results: list[ValidationResult]


class BulkTaskValidationRequest(BaseModel):
    project: str | None = None
    sprint: str | None = None
    tasks: list[TaskInput]


SYSTEM_PROMPT = """
You are Agent 1, the Task and Estimation Validation Agent for Boscosoft's Agile Project Management Tool.

Your responsibility is to review a software development task using ONLY the information provided by the user.

Your goal is NOT to redesign the task.

Your goal is to determine whether the task is sufficiently defined and whether the current estimated effort is reasonable.

Always behave as a VALIDATOR, not as a GENERATOR.

====================================================================
VALIDATION WORKFLOW
====================================================================

Perform validation in this order.

STEP 1
Validate the task description.

If the description is missing or insufficient,
STOP immediately and return the CANNOT_VALIDATE_ESTIMATE output as defined in the MANDATORY TASK DESCRIPTION section below.

Do not continue with any further validation.

----------------------------------------------------

STEP 2

Validate the task title.

----------------------------------------------------

STEP 3

Validate the task description quality.

----------------------------------------------------

STEP 4

Validate the task scope.

----------------------------------------------------

STEP 5

Validate the estimated effort.

Do NOT attempt to improve an already reasonable estimate.

Only determine whether the current estimate is reasonable.

----------------------------------------------------

STEP 6

Generate the recommendation by summarizing every assessment.

====================================================================
OUTPUT SCHEMA (MANDATORY)
====================================================================

Return EXACTLY one JSON object with EXACTLY these keys, in this order.
Do not add, remove, rename, or reorder keys. Do not nest objects. Do not use arrays.

{
  "decision": "string, one of: PROCEED | REVIEW_ESTIMATE | REWRITE_TASK | REWRITE_AND_REESTIMATE | CANNOT_VALIDATE_ESTIMATE",
  "task_title_assessment": "string, max 25 words",
  "task_description_assessment": "string, max 25 words",
  "scope_assessment": "string, max 25 words",
  "effort_assessment": "string, max 25 words",
  "suggested_task_title": "string, max 15 words",
  "suggested_task_description": "string, max 60 words",
  "suggested_estimated_hours": "number",
  "recommendation": "string, max 40 words"
}

Field-specific rules:

- "decision" must be exactly one of the five values listed above. No other text, punctuation, or explanation in this field.
- "suggested_estimated_hours" must always be a number (never a string, never null). See ESTIMATE VALIDATION and MANDATORY TASK DESCRIPTION sections for what number to use in each case.
- All string fields must be non-empty. If a section cannot be evaluated (e.g. CANNOT_VALIDATE_ESTIMATE case), use a short explanatory string such as "Not evaluated - description insufficient" rather than an empty string.
- Never return Markdown, code fences, comments, or any text outside the single JSON object.
- Use only double quotes. No trailing commas.

====================================================================
DECISION VALUES
====================================================================

Return EXACTLY one value in the "decision" field.

PROCEED

The task title, description, scope and estimated effort appear reasonable.

----------------------------------------------------

REVIEW_ESTIMATE

The task definition is acceptable but the estimated effort is clearly unrealistic.

----------------------------------------------------

REWRITE_TASK

The estimated effort appears reasonable but the title or description should be improved.

----------------------------------------------------

REWRITE_AND_REESTIMATE

Both the task definition and the estimated effort require improvement.

----------------------------------------------------

CANNOT_VALIDATE_ESTIMATE

The task description is missing or insufficient to understand the work.

====================================================================
GENERAL RULES
====================================================================

Return ONLY one JSON object, matching the OUTPUT SCHEMA exactly.

Never return Markdown.

Never use code fences.

Never include explanations outside the JSON.

Never return null.

Never return empty strings.

Every required property must always exist.

Do not create additional properties.

Use only double quotes.

Do not use trailing commas.

Use ONLY the information provided.

Never invent:

• APIs

• Database tables

• Business rules

• Frameworks

• Technologies

• Acceptance criteria

• Deployment steps

• Security requirements

• Testing requirements

• Functional requirements

If information is missing,
state that it is missing.

Never guess.

====================================================================
TASK PRIORITY
====================================================================

Priority represents ONLY business priority.

Priority DOES NOT represent:

• complexity

• effort

• implementation hours

• technical difficulty

Examples

High Priority

May require only 2 hours.

Low Priority

May require 40 hours.

Medium Priority

May require 1 hour or 80 hours.

Never estimate effort based on priority.

Estimate effort ONLY from the actual work described.

====================================================================
MANDATORY TASK DESCRIPTION
====================================================================

A meaningful task description is mandatory.

If the description is:

Empty

Whitespace

Null

Too short

Too vague

Unable to explain what work needs to be performed

Immediately return decision = CANNOT_VALIDATE_ESTIMATE with the following field values:

- "task_title_assessment": briefly note the title could not be meaningfully evaluated without a valid description.
- "task_description_assessment": explain specifically why the description is insufficient (empty, too vague, too short, etc).
- "scope_assessment": state that scope cannot be determined without a valid description.
- "effort_assessment": state that the estimate cannot be validated without a valid description.
- "suggested_task_title": return the original title unchanged. Do not invent a new title.
- "suggested_task_description": return the original description unchanged (or an empty-input placeholder such as "No description provided" if the original was truly empty/whitespace/null). Do not invent content.
- "suggested_estimated_hours": return the ORIGINAL estimated_hours value unchanged. This does NOT mean the estimate is approved or validated - it only means no alternative number can be produced without a meaningful description. The recommendation field (and the system's own message) will make clear that the estimate is UNVALIDATED, not confirmed.
- "recommendation": explicitly state that the estimate could not be validated because the description is missing or insufficient, that the originally submitted estimated hours are being carried forward unchanged only because no alternative can be computed, and that the user must provide a meaningful task description before re-validation.

Do NOT estimate effort.

Do NOT rewrite the task.

Do NOT invent details.

Do NOT suggest another estimate.

Examples of insufficient descriptions

Fix bug

API

UI

Testing

Backend

Update module

Work on page

====================================================================
TASK TITLE
====================================================================

If the title is already clear,
return it unchanged in "suggested_task_title".

Otherwise rewrite it to be

Specific

Professional

Action-oriented

Concise

Never invent information.

====================================================================
TASK DESCRIPTION
====================================================================

If the description is already understandable,
return it unchanged in "suggested_task_description".

Otherwise improve only the wording.

Do not invent any missing information.

====================================================================
SCOPE VALIDATION
====================================================================

Determine whether the scope is understandable.

Do not expand the scope.

Do not invent missing work.

====================================================================
ESTIMATE VALIDATION
====================================================================

Your responsibility is to VALIDATE the estimate.

NOT

Generate a better estimate.

Before changing the estimate ask

"Is the CURRENT estimate reasonable?"

Never ask

"What estimate would I choose?"

If the estimate is within a realistic range

Approve it.

Even if another estimate could also be reasonable.

Do NOT continuously optimize estimates.

Do NOT continuously increase estimates.

Do NOT continuously decrease estimates.

Only recommend another estimate if there is strong evidence that the current estimate is clearly unrealistic.

Examples

Reasonable differences

8 vs 10

10 vs 12

18 vs 20

12 vs 14

Do NOT change these.

Examples requiring REVIEW_ESTIMATE

1 hour for implementing authentication

2 hours for multiple APIs, UI, database and testing

80 hours for correcting one spelling mistake

40 hours for changing a button label

Only substantial differences justify changing the estimate.

====================================================================
CONSISTENCY
====================================================================

Repeated validation of the same task must produce the same result.

Example

Estimate

4

↓

Decision

REVIEW_ESTIMATE

Suggested

12

----------------------------------------------------

User updates estimate

12

Same title

Same description

Same scope

↓

Decision

PROCEED

Suggested

12

NOT

24

Never repeatedly increase estimates.

Never repeatedly decrease estimates.

Once an estimate is reasonable,

Approve it.

====================================================================
OUTPUT RULES FOR suggested_estimated_hours
====================================================================

For

PROCEED

REWRITE_TASK

Return

suggested_estimated_hours = original estimated_hours

For

REVIEW_ESTIMATE

REWRITE_AND_REESTIMATE

Return a revised estimate ONLY when the current estimate is clearly unrealistic.

For

CANNOT_VALIDATE_ESTIMATE

Return

suggested_estimated_hours = original estimated_hours, unchanged (this is NOT an approval of the estimate - it is carried forward only because no alternative can be computed without a valid description; see MANDATORY TASK DESCRIPTION section)

====================================================================
ASSESSMENT RULES
====================================================================

Each assessment should explain WHY.

Maximum 25 words.

task_title_assessment

Evaluate title clarity.

task_description_assessment

Evaluate description completeness.

scope_assessment

Evaluate scope clarity.

effort_assessment

Evaluate whether the estimate appears reasonable.

Never contradict another assessment.

====================================================================
RECOMMENDATION
====================================================================

The recommendation must summarize ALL validation results.

Never provide a generic recommendation.

Summarize

• Task title

• Task description

• Scope

• Estimated effort

If changes were suggested,

explicitly mention every change.

Examples

Everything acceptable

"The task title, description, scope, and estimated effort appear reasonable. The task is ready for implementation."

----------------------------------------------------

Only estimate changed

"The task title, description, and scope appear reasonable. The estimated effort was revised because the original estimate appears too low."

----------------------------------------------------

Only title changed

"The task title was improved for clarity. The description, scope, and estimated effort appear reasonable."

----------------------------------------------------

Title and description changed

"The task title and description were improved for clarity. The estimated effort appears reasonable."

----------------------------------------------------

Everything changed

"The task title and description were improved for clarity. The estimated effort was revised because the original estimate appears unrealistic."

----------------------------------------------------

Description insufficient (CANNOT_VALIDATE_ESTIMATE)

"The estimate could not be validated because the task description is missing or insufficient. The originally submitted estimate is carried forward unchanged, not approved. Please provide a meaningful description and resubmit for validation."

The recommendation MUST always agree with the assessment fields and with the decision value.

====================================================================
WRITING STYLE
====================================================================

Use concise professional language.

Maximum lengths

task_title_assessment

25 words

task_description_assessment

25 words

scope_assessment

25 words

effort_assessment

25 words

suggested_task_title

15 words

suggested_task_description

60 words

recommendation

40 words

Use cautious wording

appears reasonable

may require clarification

appears too low

appears too high

insufficient information

Never present opinions as facts.

====================================================================
FINAL INSTRUCTION
====================================================================

You are an ESTIMATION VALIDATOR.

Your objective is NOT to improve estimates.

Your objective is to determine whether the CURRENT estimate is reasonable.

Never continuously optimize estimates.

Never repeatedly change an already reasonable estimate.

Priority is ONLY business priority.

Priority is NOT complexity.

Never estimate effort based on priority.

Never invent missing information.

Never assume missing requirements.

A meaningful task description is mandatory.

Without a meaningful task description,

always return decision = CANNOT_VALIDATE_ESTIMATE with suggested_estimated_hours = the original estimated_hours value, unchanged, per the MANDATORY TASK DESCRIPTION section.

Human review is always required before applying AI recommendations.

Always return output matching the OUTPUT SCHEMA exactly — same keys, same order, no additions, no omissions.
"""


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() == "nan":
        return ""

    return re.sub(r"\s+", " ", text)


def truncate_for_prompt(
    text: str,
    max_chars: int = MAX_DESCRIPTION_CHARS,
) -> str:
    """Cap long free-text fields before they go into a Groq prompt.

    Protects the per-minute token budget from a single outsized task
    description; does not affect what is stored/returned, only what is
    sent to the model.
    """
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " …[truncated for length]"


def normalize_header(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


COLUMN_ALIASES = {
    "task_id": {"task id", "taskid"},
    "module_name": {"module name", "modulename", "module"},
    "feature_name": {"feature name", "featurename", "feature"},
    "task_title": {
        "task title",
        "tasktitle",
        "task name",
        "taskname",
        "task",
    },
    "task_description": {
        "task description",
        "taskdescription",
        "description",
    },
    "estimated_hours": {
        "estimate hrs",
        "estimatehrs",
        "estimated hours",
        "estimated hrs",
        "est hrs",
        "est. hrs",
        "planned hours",
        "planned_hours",
    },
    "complexity": {"complexity"},
    "project_tag": {
        "project tag",
        "projecttag",
        "project",
    },
    "assignee": {
        "assign to",
        "assigned to",
        "assignee",
        "employee name",
        "employeename",
    },
}


def get_value(
    row: dict[str, Any],
    logical_field: str,
) -> Any:
    aliases = COLUMN_ALIASES[logical_field]

    for column, value in row.items():
        if normalize_header(column) in aliases:
            return value

    return ""


def hours_to_decimal(value: Any) -> float:
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        number = float(value)

        if pd.isna(number):
            return 0.0

        if 0 < number < 1:
            return round(number * 24, 2)

        return round(number, 2)

    text = clean_text(value).lower()

    if not text:
        return 0.0

    match = re.fullmatch(
        r"(\d+):(\d{1,2})(?::(\d{1,2}))?",
        text,
    )

    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3) or 0)

        return round(
            hours + minutes / 60 + seconds / 3600,
            2,
        )

    number_match = re.search(r"\d+(?:\.\d+)?", text)

    if number_match:
        return float(number_match.group())

    return 0.0


def build_prompt(
    task: TaskInput,
    project: str | None = None,
    sprint: str | None = None,
) -> str:
    validation_context = {
        "project": clean_text(project),
        "sprint": clean_text(sprint),
    }

    # Only the prompt payload is truncated for token-budget reasons; the
    # ValidationResult returned to the caller always uses the untouched
    # original text (see normalize_validation_result / TaskInput).
    prompt_description = truncate_for_prompt(task.task_description)
    prompt_task = task.model_copy(
        update={"task_description": prompt_description}
    )

    return f"""
Evaluate this backlog task:

{prompt_task.model_dump_json(indent=2)}

Additional validation context:
{json.dumps(validation_context, indent=2)}

Return only JSON in this exact structure:

{{
  "task_id": "{task.task_id}",
  "decision": "PROCEED | REVIEW_ESTIMATE | REWRITE_TASK | REWRITE_AND_REESTIMATE | CANNOT_VALIDATE_ESTIMATE",
  "task_title_assessment": "A clear assessment of the original title",
  "task_description_assessment": "A clear assessment of the original description",
  "scope_assessment": "An assessment of whether the scope is appropriate",
  "effort_assessment": "An assessment of whether the original estimate is reasonable",
  "suggested_task_title": "Always return the original title or a corrected title",
  "suggested_task_description": "Always return the original description or a corrected description",
  "suggested_estimated_hours": {task.estimated_hours},
  "confidence_score": 0.0,
  "recommendation": "A concise recommendation for human review"
}}

Important:
- If the original title is correct, return exactly: {json.dumps(task.task_title)}
- If the original description is correct, return exactly: {json.dumps(prompt_description)}
- If task_description above was truncated for length, do not treat the
  truncation marker as part of the task's actual scope.
- If decision is PROCEED, REWRITE_TASK, or CANNOT_VALIDATE_ESTIMATE,
  return suggested_estimated_hours exactly as: {task.estimated_hours}
- If decision is REVIEW_ESTIMATE or REWRITE_AND_REESTIMATE,
  return a realistic revised numeric estimate that is different from: {task.estimated_hours}
- Do not return null, an empty string, 'No changes required', or 'Not applicable'.
"""


def normalize_validation_result(
    task: TaskInput,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    """Guarantee that all suggestion fields contain usable values."""
    parsed["task_id"] = task.task_id

    suggested_title = clean_text(parsed.get("suggested_task_title"))
    suggested_description = clean_text(
        parsed.get("suggested_task_description")
    )

    if not suggested_title or suggested_title.lower() in {
        "null",
        "none",
        "no changes required",
        "not applicable",
        "n/a",
    }:
        parsed["suggested_task_title"] = task.task_title
    else:
        parsed["suggested_task_title"] = suggested_title

    if not suggested_description or suggested_description.lower() in {
        "null",
        "none",
        "no changes required",
        "not applicable",
        "n/a",
    }:
        parsed["suggested_task_description"] = task.task_description
    else:
        parsed["suggested_task_description"] = suggested_description

    suggested_hours = parsed.get("suggested_estimated_hours")
    try:
        parsed["suggested_estimated_hours"] = float(suggested_hours)
    except (TypeError, ValueError):
        parsed["suggested_estimated_hours"] = task.estimated_hours

    try:
        confidence = float(parsed.get("confidence_score", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    parsed["confidence_score"] = min(max(confidence, 0.0), 1.0)

    for field_name in (
        "decision",
        "task_title_assessment",
        "task_description_assessment",
        "scope_assessment",
        "effort_assessment",
        "recommendation",
    ):
        parsed[field_name] = clean_text(parsed.get(field_name))

    return parsed


def build_suggestion_summary(
    task: TaskInput,
    parsed: dict[str, Any],
) -> str:
    """Deterministically state every suggested change.

    Requirement: "All suggested information must also be in the
    Recommendation." The system prompt asks the model to summarize its
    own suggestions, but that is not guaranteed - the model can drift,
    be vague, or omit a field. This function is a code-level guarantee:
    it inspects the actual suggested_task_title, suggested_task_description
    and suggested_estimated_hours values and appends a factual, literal
    statement of each one that differs from the original, so the
    recommendation always reflects the real suggested values regardless
    of what the model wrote.
    """
    decision = parsed.get("decision", "")

    if decision == "CANNOT_VALIDATE_ESTIMATE":
        return (
            "No suggested changes are available: the estimate could not "
            "be validated, so the original title, description, and "
            "estimated hours are carried forward unchanged (not approved)."
        )

    parts: list[str] = []

    suggested_title = clean_text(
        parsed.get("suggested_task_title", task.task_title)
    )
    if suggested_title != clean_text(task.task_title):
        parts.append(f'Suggested task title: "{suggested_title}".')

    suggested_description = clean_text(
        parsed.get("suggested_task_description", task.task_description)
    )
    if suggested_description != clean_text(task.task_description):
        parts.append(
            f'Suggested task description: "{suggested_description}".'
        )

    try:
        suggested_hours = float(
            parsed.get("suggested_estimated_hours", task.estimated_hours)
        )
    except (TypeError, ValueError):
        suggested_hours = task.estimated_hours

    if not hours_are_equal(suggested_hours, task.estimated_hours):
        parts.append(
            f"Suggested estimated hours: {suggested_hours} "
            f"(originally {task.estimated_hours})."
        )

    if not parts:
        return (
            "No changes suggested: the original title, description, and "
            "estimated hours all appear reasonable and remain unchanged."
        )

    return " ".join(parts)


def validation_error_result(
    task: TaskInput,
    error: Exception,
) -> ValidationResult:
    return ValidationResult(
        task_id=task.task_id,
        decision="ERROR",
        task_title_assessment="Validation could not be completed.",
        task_description_assessment="Validation could not be completed.",
        scope_assessment="Validation could not be completed.",
        effort_assessment="Validation could not be completed.",
        suggested_task_title=task.task_title,
        suggested_task_description=task.task_description,
        suggested_estimated_hours=task.estimated_hours,
        confidence_score=0.0,
        recommendation=f"Validation failed: {error}",
    )



def hours_are_equal(
    first: float,
    second: float,
    tolerance: float = 0.01,
) -> bool:
    return abs(first - second) <= tolerance


# ==============================================================
# VALIDATION RESULT CACHE
# ==============================================================
# Requirement: re-validating the same task, unchanged, must ALWAYS
# produce the same suggested estimate - not "usually", not "until the
# process restarts". These estimates get assigned to developers and
# shown to clients, so they must be exact and stable.
#
# temperature=0 and the prompt's CONSISTENCY section make the model
# *likely* to be consistent, but hosted LLM sampling does not strictly
# guarantee identical output across separate calls. So instead of
# hoping the model agrees with itself, every result is cached by a
# fingerprint of the task's content: an unchanged task never calls the
# LLM twice and therefore cannot get a different answer. Any real
# change to any field (title, description, hours, complexity, module,
# feature, project tag) yields a different fingerprint and correctly
# triggers fresh validation.
#
# Persistence: this cache is backed by Redis (REDIS_URL) when it's
# configured and reachable, so a result survives container restarts
# and is shared across every replica/worker of this service - a
# result computed on one instance is reused by all the others instead
# of each one silently deciding for itself. If Redis is not configured
# or is temporarily unreachable, the service "fails open": it falls
# back to a process-local in-memory cache rather than refusing to
# validate. That local fallback is only exact within a single running
# process - point Redis at a real, reachable instance (docker-compose
# already provisions one) whenever more than one instance/worker of
# this service is running, or whenever restarts must not reset it.
#
# VALIDATION_CACHE_TTL_SECONDS controls storage retention only, not
# correctness: 0 (default) means cached results never expire on their
# own. Freshness is never time-based - a task is only ever re-evaluated
# because something in it actually changed (a different fingerprint),
# never because a clock ran out.

CACHE_KEY_PREFIX = "agent1:validation:"

REDIS_URL = os.getenv("REDIS_URL", "").strip()
VALIDATION_CACHE_TTL_SECONDS = int(
    os.getenv("VALIDATION_CACHE_TTL_SECONDS", "0")
)

try:
    import redis as _redis_module
except ImportError:
    _redis_module = None
    logger.warning(
        "The 'redis' package is not installed; the validation cache "
        "will be in-memory only for this process (see requirements.txt)."
    )

_redis_client = None
if REDIS_URL and _redis_module is not None:
    try:
        _redis_client = _redis_module.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        _redis_client.ping()
        logger.info(
            "Validation cache backed by Redis at %s - results persist "
            "across restarts and are shared across replicas.",
            REDIS_URL,
        )
    except Exception as exc:
        logger.warning(
            "Redis at %s is unreachable (%s); failing open to an "
            "in-memory-only validation cache for this process. Cached "
            "results will NOT survive a restart or be shared across "
            "other instances until Redis is reachable.",
            REDIS_URL,
            exc,
        )
        _redis_client = None
elif not REDIS_URL:
    logger.info(
        "REDIS_URL is not set; the validation cache is in-memory only "
        "for this process. Set REDIS_URL (docker-compose already "
        "provides one) so results persist across restarts and are "
        "shared across replicas."
    )

# Always-present fallback layer, used when Redis is absent/unreachable
# and as a fast local mirror when Redis IS present.
_validation_cache: dict[str, ValidationResult] = {}
_validation_cache_lock = threading.Lock()


def compute_task_fingerprint(
    task: TaskInput,
    project: str | None = None,
    sprint: str | None = None,
) -> str:
    """Stable fingerprint of everything that can influence validation."""
    payload = {
        "task_id": task.task_id,
        "module_name": clean_text(task.module_name),
        "feature_name": clean_text(task.feature_name),
        "task_title": clean_text(task.task_title),
        "task_description": clean_text(task.task_description),
        "estimated_hours": round(task.estimated_hours, 2),
        "complexity": clean_text(task.complexity).lower(),
        "project_tag": clean_text(task.project_tag),
        "project": clean_text(project),
        "sprint": clean_text(sprint),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_cached_result(fingerprint: str) -> ValidationResult | None:
    if _redis_client is not None:
        try:
            raw = _redis_client.get(CACHE_KEY_PREFIX + fingerprint)
            if raw is not None:
                return ValidationResult.model_validate_json(raw)
            return None
        except Exception as exc:
            logger.warning(
                "Redis GET failed (%s); falling back to the in-memory "
                "cache for this lookup.",
                exc,
            )

    with _validation_cache_lock:
        cached = _validation_cache.get(fingerprint)
    return cached.model_copy() if cached is not None else None


def store_cached_result(
    fingerprint: str,
    result: ValidationResult,
) -> None:
    # Never cache ERROR results: a transient failure (timeout, rate
    # limit exhaustion, malformed model output, etc.) should not
    # permanently "stick" - the next validation attempt for the same
    # task should be free to try the LLM again.
    if result.decision == "ERROR":
        return

    if _redis_client is not None:
        try:
            payload = result.model_dump_json()
            key = CACHE_KEY_PREFIX + fingerprint
            if VALIDATION_CACHE_TTL_SECONDS > 0:
                _redis_client.set(
                    key, payload, ex=VALIDATION_CACHE_TTL_SECONDS
                )
            else:
                _redis_client.set(key, payload)
        except Exception as exc:
            logger.warning(
                "Redis SET failed (%s); this result is only cached "
                "in-memory on this process for now.",
                exc,
            )

    # Always keep a local copy too: it's what serves reads if Redis is
    # momentarily unreachable, and it's the only copy at all when
    # Redis isn't configured.
    with _validation_cache_lock:
        _validation_cache[fingerprint] = result.model_copy()


def clear_validation_cache() -> None:
    if _redis_client is not None:
        try:
            for prefix in (CACHE_KEY_PREFIX, CANONICAL_KEY_PREFIX):
                cursor = 0
                while True:
                    cursor, keys = _redis_client.scan(
                        cursor=cursor,
                        match=prefix + "*",
                        count=500,
                    )
                    if keys:
                        _redis_client.delete(*keys)
                    if cursor == 0:
                        break
        except Exception as exc:
            logger.warning(
                "Redis cache clear failed (%s); in-memory cache was "
                "still cleared.",
                exc,
            )

    with _validation_cache_lock:
        _validation_cache.clear()
        _canonical_cache.clear()


# ==============================================================
# CANONICAL ESTIMATE
# ==============================================================
# The exact-fingerprint cache above only guarantees that literally
# resubmitting the SAME estimated_hours for the SAME task returns the
# SAME answer. It does NOT stop the LLM from picking a DIFFERENT
# "correct" number each time it's asked to judge a DIFFERENT
# estimated_hours against the same task - e.g. submitting 8h might get
# corrected to "48h is realistic", and later submitting 40h for the
# exact same task/description might get corrected to "50h is
# realistic" instead of also landing on 48h. Both individual answers
# can look plausible to the model in isolation, but together they
# contradict each other - the task's real effort cannot simultaneously
# be 48h and 50h when nothing about the task itself changed.
#
# These estimates are assigned to developers and shown to clients, so
# they must be exact and stable - not "the model's best guess this
# particular time". So: the FIRST time a task's content (title,
# description, scope, complexity - everything EXCEPT the estimated
# hours being tested) is ever validated, whatever number the model
# lands on for "what this task should realistically take" is PINNED as
# that task's one canonical estimate. Every later validation of the
# same content - no matter what estimated_hours is submitted, and no
# matter how many times - reuses that same pinned number. No further
# LLM call is made for those checks at all: the decision becomes a
# plain, deterministic comparison in code between the submitted hours
# and the pinned canonical hours. That is what actually guarantees
# exactness, rather than hoping the model agrees with itself.
#
# The canonical estimate is only ever replaced when the task's own
# content changes (a different content fingerprint) - never because a
# different number was tested against it.

CANONICAL_KEY_PREFIX = "agent1:canonical:"

# How close a submitted estimate must be to the canonical estimate to
# be accepted without being flagged. Accepts the LARGER of a flat hour
# allowance and a percentage of the canonical estimate, so small tasks
# keep a sane minimum cushion and large tasks scale sensibly. Tune via
# env if your organization's tolerance for "close enough" differs.
ESTIMATE_TOLERANCE_ABS_HOURS = float(
    os.getenv("ESTIMATE_TOLERANCE_ABS_HOURS", "2")
)
ESTIMATE_TOLERANCE_PCT = float(
    os.getenv("ESTIMATE_TOLERANCE_PCT", "0.20")
)

_canonical_cache: dict[str, dict[str, Any]] = {}


def compute_content_fingerprint(
    task: TaskInput,
    project: str | None = None,
    sprint: str | None = None,
) -> str:
    """Fingerprint of everything EXCEPT estimated_hours.

    Identifies "the same task" for pinning a canonical estimate: two
    validations with the same title/description/scope/complexity are
    the same task even when a different estimated_hours is being
    tested against it.
    """
    payload = {
        "task_id": task.task_id,
        "module_name": clean_text(task.module_name),
        "feature_name": clean_text(task.feature_name),
        "task_title": clean_text(task.task_title),
        "task_description": clean_text(task.task_description),
        "complexity": clean_text(task.complexity).lower(),
        "project_tag": clean_text(task.project_tag),
        "project": clean_text(project),
        "sprint": clean_text(sprint),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def estimate_tolerance(canonical_hours: float) -> float:
    return max(
        ESTIMATE_TOLERANCE_ABS_HOURS,
        ESTIMATE_TOLERANCE_PCT * canonical_hours,
    )


def get_canonical_estimate(
    content_fingerprint: str,
) -> dict[str, Any] | None:
    key = CANONICAL_KEY_PREFIX + content_fingerprint
    if _redis_client is not None:
        try:
            raw = _redis_client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:
            logger.warning(
                "Redis GET failed for canonical estimate (%s); "
                "falling back to the in-memory copy for this lookup.",
                exc,
            )

    with _validation_cache_lock:
        canonical = _canonical_cache.get(content_fingerprint)
    return dict(canonical) if canonical is not None else None


def store_canonical_estimate(
    content_fingerprint: str,
    canonical: dict[str, Any],
) -> None:
    key = CANONICAL_KEY_PREFIX + content_fingerprint
    if _redis_client is not None:
        try:
            payload = json.dumps(canonical)
            if VALIDATION_CACHE_TTL_SECONDS > 0:
                _redis_client.set(
                    key, payload, ex=VALIDATION_CACHE_TTL_SECONDS
                )
            else:
                _redis_client.set(key, payload)
        except Exception as exc:
            logger.warning(
                "Redis SET failed for canonical estimate (%s); this "
                "task's canonical estimate is only pinned in-memory on "
                "this process for now.",
                exc,
            )

    with _validation_cache_lock:
        _canonical_cache[content_fingerprint] = dict(canonical)


def build_result_from_canonical(
    task: TaskInput,
    canonical: dict[str, Any],
) -> ValidationResult:
    """Deterministically compare the submitted hours to the pinned
    canonical estimate - no LLM call, so no room for a different
    "right answer" to be invented this time around.
    """
    canonical_hours = float(canonical["canonical_hours"])
    tolerance = estimate_tolerance(canonical_hours)
    hours_ok = abs(task.estimated_hours - canonical_hours) <= tolerance

    suggested_title = canonical.get(
        "suggested_task_title", task.task_title
    )
    suggested_description = canonical.get(
        "suggested_task_description", task.task_description
    )
    needs_rewrite = (
        clean_text(suggested_title) != clean_text(task.task_title)
        or clean_text(suggested_description)
        != clean_text(task.task_description)
    )

    if needs_rewrite and not hours_ok:
        decision = "REWRITE_AND_REESTIMATE"
    elif needs_rewrite and hours_ok:
        decision = "REWRITE_TASK"
    elif not hours_ok:
        decision = "REVIEW_ESTIMATE"
    else:
        decision = "PROCEED"

    suggested_hours = task.estimated_hours if hours_ok else canonical_hours

    if hours_ok:
        effort_assessment = (
            f"The submitted estimate ({task.estimated_hours}h) is "
            f"consistent with the validated effort already established "
            f"for this task ({canonical_hours}h); no change needed."
        )
        recommendation = (
            "The task title, description, and scope were already "
            "validated for this task and remain unchanged. The "
            "submitted estimate is consistent with the previously "
            "validated effort."
        )
    else:
        effort_assessment = (
            f"The submitted estimate ({task.estimated_hours}h) differs "
            f"substantially from the effort previously validated for "
            f"this task ({canonical_hours}h)."
        )
        recommendation = (
            "The task title, description, and scope were already "
            "validated for this task and remain unchanged. The "
            "submitted estimate does not match the previously "
            "validated effort and should be revised."
        )

    parsed: dict[str, Any] = {
        "task_id": task.task_id,
        "decision": decision,
        "task_title_assessment": canonical.get(
            "task_title_assessment",
            "Not re-evaluated - title unchanged since it was last validated.",
        ),
        "task_description_assessment": canonical.get(
            "task_description_assessment",
            "Not re-evaluated - description unchanged since it was last validated.",
        ),
        "scope_assessment": canonical.get(
            "scope_assessment",
            "Not re-evaluated - scope unchanged since it was last validated.",
        ),
        "effort_assessment": effort_assessment,
        "suggested_task_title": suggested_title,
        "suggested_task_description": suggested_description,
        "suggested_estimated_hours": suggested_hours,
        "confidence_score": canonical.get("confidence_score", 0.8),
        "recommendation": recommendation,
    }

    suggestion_summary = build_suggestion_summary(task, parsed)
    if (
        suggestion_summary
        and suggestion_summary.lower() not in recommendation.lower()
    ):
        recommendation = f"{recommendation} {suggestion_summary}".strip()

    disclaimer = (
        "This is an AI-generated estimation review based only on "
        "the supplied task information; human review is required before applying any changes."
    )
    if disclaimer.lower() not in recommendation.lower():
        recommendation = f"{recommendation} {disclaimer}".strip()

    parsed["recommendation"] = recommendation

    return ValidationResult.model_validate(parsed)


# ==============================================================
# GROQ ERROR TAXONOMY
# ==============================================================
# Mirrors Agent 2's classification so both services fail the same way:
# auth errors never retry, rate limits back off and retry, 5xx/timeouts/
# connection errors retry, and other 4xx errors fail immediately.


class GroqAuthenticationFailure(Exception):
    """The Groq API rejected our credentials. Never retryable."""


class GroqRateLimitFailure(Exception):
    """HTTP 429 - retryable, but should back off longer than a generic
    transient failure since it means the per-minute budget is exhausted."""


class GroqTransientFailure(Exception):
    """Timeout, connection, or 5xx error from Groq. Safe to retry."""


class GroqPermanentFailure(Exception):
    """Any other non-retryable Groq API error (e.g. a 4xx bad request)."""


def _is_json_validate_failed(exc: APIStatusError) -> bool:
    """True if Groq's response body reports code == "json_validate_failed".

    This is a 400 from Groq's JSON-mode/structured-output validator when
    the model's generated text failed to parse or match the requested
    shape. It is a sampling hiccup, not a malformed request - the same
    prompt commonly succeeds on retry - so it is treated as transient
    rather than permanent, unlike other 4xx errors.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = body.get("code") or body.get("error", {}).get("code")
        if code == "json_validate_failed":
            return True
    return "json_validate_failed" in str(exc)


def call_groq_llm(user_prompt: str) -> str:
    """Call the Groq chat completion API and return the raw text response.

    Bounded by GROQ_MAX_CONCURRENT_REQUESTS so a bulk validation batch
    never has more than that many requests in flight against Groq at
    once, independent of how many threads the executor is running.

    Raises:
        GroqAuthenticationFailure: invalid/missing credentials.
        GroqRateLimitFailure: HTTP 429 - caller should back off and retry.
        GroqTransientFailure: timeout, connection, or 5xx error - safe to
            retry.
        GroqPermanentFailure: any other non-retryable API error.
    """
    with _groq_concurrency_gate:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0,
                max_completion_tokens=1200,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""

        except AuthenticationError as exc:
            logger.error("Groq authentication error: %s", exc)
            raise GroqAuthenticationFailure(str(exc)) from exc

        except RateLimitError as exc:
            logger.warning("Groq rate limit (429) hit: %s", exc)
            raise GroqRateLimitFailure(str(exc)) from exc

        except APITimeoutError as exc:
            logger.warning("Groq request timed out: %s", exc)
            raise GroqTransientFailure(str(exc)) from exc

        except APIConnectionError as exc:
            logger.warning("Groq connection error: %s", exc)
            raise GroqTransientFailure(str(exc)) from exc

        except APIStatusError as exc:
            if exc.status_code >= 500:
                logger.warning(
                    "Groq API returned a server error: %s", exc
                )
                raise GroqTransientFailure(str(exc)) from exc
            if exc.status_code == 400 and _is_json_validate_failed(exc):
                logger.warning(
                    "Groq JSON validation failed (400 "
                    "json_validate_failed), treating as retryable: %s",
                    exc,
                )
                raise GroqTransientFailure(str(exc)) from exc
            logger.error(
                "Groq API returned a client error: %s", exc
            )
            raise GroqPermanentFailure(str(exc)) from exc


def request_groq_validation(
    task: TaskInput,
    user_prompt: str,
) -> dict[str, Any]:
    """Call Groq and parse its JSON response, retrying on transient
    failures (including rate limits) with exponential backoff and jitter.

    Auth failures and other non-retryable (4xx) API errors are raised
    immediately without retrying, matching Agent 2's behavior.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            raw = call_groq_llm(user_prompt)

            if not raw:
                raise RuntimeError("Groq returned an empty response")

            parsed = json.loads(raw)

            if not isinstance(parsed, dict):
                raise RuntimeError(
                    "Groq response must be a JSON object"
                )

            return parsed

        except GroqAuthenticationFailure:
            raise

        except GroqPermanentFailure:
            raise

        except GroqRateLimitFailure as exc:
            last_error = exc
            # Rate limits get a longer, exponentially growing wait than
            # other transient errors, since retrying immediately just
            # burns the retry budget against a still-full quota.
            wait_seconds = (
                RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                + random.uniform(0, 0.5)
            )
            logger.warning(
                "Rate limited (attempt %d/%d); backing off %.1fs",
                attempt,
                MAX_LLM_RETRIES,
                wait_seconds,
            )
            if attempt < MAX_LLM_RETRIES:
                time.sleep(wait_seconds)

        except (
            GroqTransientFailure,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            logger.warning(
                "Groq attempt %d/%d failed (%s): %s",
                attempt,
                MAX_LLM_RETRIES,
                type(exc).__name__,
                exc,
            )
            if attempt < MAX_LLM_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Groq validation failed after {MAX_LLM_RETRIES} attempts: "
        f"{last_error}"
    )


def build_estimate_correction_prompt(
    task: TaskInput,
    parsed: dict[str, Any],
    project: str | None = None,
    sprint: str | None = None,
) -> str:
    return f"""
The previous validation response is inconsistent.

Original task:
{task.model_dump_json(indent=2)}

Validation context:
{json.dumps({"project": clean_text(project), "sprint": clean_text(sprint)}, indent=2)}

Previous response:
{json.dumps(parsed, indent=2)}

The decision is {parsed.get("decision")}, so suggested_estimated_hours
must be a realistic numeric estimate different from the original
estimated_hours value of {task.estimated_hours}.

Re-evaluate the effort using the task scope, complexity, expected
activities, and functional boundary. Keep all other output fields
complete and return only valid JSON in the same structure.
"""

def validate_with_groq(
    task: TaskInput,
    project: str | None = None,
    sprint: str | None = None,
) -> ValidationResult:
    fingerprint = compute_task_fingerprint(task, project, sprint)

    cached = get_cached_result(fingerprint)
    if cached is not None:
        logger.info(
            "Task %s unchanged since last validation; returning the "
            "same cached result instead of calling the LLM again.",
            task.task_id,
        )
        # task_id is echoed from the current request even on a cache
        # hit, in case the same content was previously validated under
        # a different task_id.
        return cached.model_copy(update={"task_id": task.task_id})

    # A different estimated_hours than last time means a cache miss
    # above, but that does NOT mean a different "right answer" should
    # be invented. If this task's content has already produced a
    # pinned canonical estimate, compare against that instead of
    # asking the LLM to judge a fresh target - see CANONICAL ESTIMATE.
    content_fingerprint = compute_content_fingerprint(task, project, sprint)
    canonical = get_canonical_estimate(content_fingerprint)

    if canonical is not None:
        logger.info(
            "Task %s: reusing the previously pinned canonical estimate "
            "(%sh) for this task instead of asking the LLM to judge a "
            "new target.",
            task.task_id,
            canonical.get("canonical_hours"),
        )
        result = build_result_from_canonical(task, canonical)
        store_cached_result(fingerprint, result)
        return result

    try:
        parsed = request_groq_validation(
            task,
            build_prompt(task, project, sprint),
        )
        parsed = normalize_validation_result(task, parsed)

        decision = parsed.get("decision", "").upper()
        parsed["decision"] = decision

        estimate_review_decisions = {
            "REVIEW_ESTIMATE",
            "REWRITE_AND_REESTIMATE",
        }

        if (
            decision in estimate_review_decisions
            and hours_are_equal(
                float(parsed["suggested_estimated_hours"]),
                task.estimated_hours,
            )
        ):
            parsed = request_groq_validation(
                task,
                build_estimate_correction_prompt(
                    task,
                    parsed,
                    project,
                    sprint,
                ),
            )
            parsed = normalize_validation_result(task, parsed)

            corrected_decision = parsed.get(
                "decision",
                "",
            ).upper()
            parsed["decision"] = corrected_decision

            if (
                corrected_decision in estimate_review_decisions
                and hours_are_equal(
                    float(parsed["suggested_estimated_hours"]),
                    task.estimated_hours,
                )
            ):
                raise RuntimeError(
                    "The model marked the estimate for review "
                    "but did not provide a revised estimate."
                )

        final_decision = parsed.get("decision", "").upper()
        parsed["decision"] = final_decision

        non_estimate_change_decisions = {
            "PROCEED",
            "REWRITE_TASK",
            "CANNOT_VALIDATE_ESTIMATE",
        }

        if final_decision in non_estimate_change_decisions:
            parsed["suggested_estimated_hours"] = (
                task.estimated_hours
            )

        recommendation = clean_text(parsed.get("recommendation"))

        # Requirement: every suggested change must appear in the
        # recommendation. Built from the actual field values rather than
        # trusted from the model, so this is guaranteed rather than
        # merely requested via the prompt.
        suggestion_summary = build_suggestion_summary(task, parsed)
        if suggestion_summary and suggestion_summary.lower() not in recommendation.lower():
            recommendation = f"{recommendation} {suggestion_summary}".strip()

        disclaimer = (
            "This is an AI-generated estimation review based only on "
            "the supplied task information; human review is required before applying any changes."
        )
        if disclaimer.lower() not in recommendation.lower():
            recommendation = f"{recommendation} {disclaimer}".strip()
        parsed["recommendation"] = recommendation

        result = ValidationResult.model_validate(parsed)

        # Pin this task's content to the estimate the model just
        # produced, so every future check - regardless of what
        # estimated_hours gets tested against it - compares against
        # this same number instead of letting the LLM invent a new
        # one. Skipped for CANNOT_VALIDATE_ESTIMATE: there is no real
        # target to pin without a valid description.
        if result.decision != "CANNOT_VALIDATE_ESTIMATE":
            canonical_hours = (
                task.estimated_hours
                if result.decision in {"PROCEED", "REWRITE_TASK"}
                else result.suggested_estimated_hours
            )
            store_canonical_estimate(
                content_fingerprint,
                {
                    "canonical_hours": canonical_hours,
                    "task_title_assessment": result.task_title_assessment,
                    "task_description_assessment": result.task_description_assessment,
                    "scope_assessment": result.scope_assessment,
                    "suggested_task_title": result.suggested_task_title,
                    "suggested_task_description": result.suggested_task_description,
                    "confidence_score": result.confidence_score,
                },
            )

        store_cached_result(fingerprint, result)
        return result

    except GroqAuthenticationFailure as error:
        # The API key itself is bad - this isn't a per-task problem, so
        # it should surface as a hard failure rather than a quiet
        # per-task ERROR result that looks like a content issue.
        raise HTTPException(
            status_code=401,
            detail="Authentication with the Groq API failed. Check GROQ_API_KEY.",
        ) from error

    except Exception as error:
        return validation_error_result(task, error)



def validate_tasks_concurrently(
    tasks: list[TaskInput],
    project: str | None = None,
    sprint: str | None = None,
) -> list[ValidationResult]:
    """Validate tasks concurrently while preserving the input order.

    If the Groq API key itself is invalid, every task would fail
    identically, so validation stops at the first authentication failure
    (401) instead of burning further Groq calls/retries against a dead
    key. Any other per-task failure still degrades to an ERROR result for
    that task only, leaving the rest of the batch unaffected.
    """
    if not tasks:
        return []

    worker_count = min(BULK_MAX_WORKERS, len(tasks))
    ordered_results: list[ValidationResult | None] = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(
                validate_with_groq,
                task,
                project,
                sprint,
            ): index
            for index, task in enumerate(tasks)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            task = tasks[index]
            try:
                ordered_results[index] = future.result()
            except HTTPException:
                for pending in future_to_index:
                    pending.cancel()
                raise
            except Exception as error:
                ordered_results[index] = validation_error_result(
                    task,
                    error,
                )

    return [
        result
        for result in ordered_results
        if result is not None
    ]

def dataframe_to_tasks(
    dataframe: pd.DataFrame,
) -> list[TaskInput]:
    tasks: list[TaskInput] = []

    for index, row in dataframe.iterrows():
        record = row.to_dict()

        task_title = clean_text(
            get_value(record, "task_title")
        )

        if not task_title:
            continue

        task_id = clean_text(
            get_value(record, "task_id")
        )

        if not task_id:
            task_id = f"ROW-{index + 2}"

        tasks.append(
            TaskInput(
                task_id=task_id,
                module_name=clean_text(
                    get_value(record, "module_name")
                ),
                feature_name=clean_text(
                    get_value(record, "feature_name")
                ),
                task_title=task_title,
                task_description=clean_text(
                    get_value(
                        record,
                        "task_description",
                    )
                ),
                estimated_hours=hours_to_decimal(
                    get_value(
                        record,
                        "estimated_hours",
                    )
                ),
                complexity=clean_text(
                    get_value(record, "complexity")
                ),
                project_tag=clean_text(
                    get_value(record, "project_tag")
                ),
                assignee=clean_text(
                    get_value(record, "assignee")
                ),
            )
        )

    return tasks


def build_bulk_response(
    results: list[ValidationResult],
) -> BulkResponse:
    failed = sum(
        result.decision == "ERROR"
        for result in results
    )
    validated = len(results) - failed

    if failed == 0:
        status = "COMPLETED"
    elif validated == 0:
        status = "FAILED"
    else:
        status = "PARTIALLY_COMPLETED"

    return BulkResponse(
        status=status,
        total_tasks=len(results),
        validated_tasks=validated,
        failed_tasks=failed,
        results=results,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": GROQ_MODEL,
    }


@app.get("/health/live")
def health_live() -> dict[str, str]:
    """Liveness probe: is the process itself up and serving requests.

    Deliberately does not check downstream dependencies (Groq, Redis) -
    that's what /health/ready is for. The Dockerfile's HEALTHCHECK
    polls this exact path; it previously had nothing to hit here,
    which meant every container was reported unhealthy on a 30s cycle
    regardless of whether anything was actually wrong.
    """
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready() -> dict[str, Any]:
    """Readiness probe: is this instance ready to serve real traffic."""
    redis_status = "not_configured"
    if REDIS_URL:
        try:
            if _redis_client is not None:
                _redis_client.ping()
                redis_status = "connected"
            else:
                redis_status = "unreachable"
        except Exception:
            redis_status = "unreachable"

    groq_configured = bool(GROQ_API_KEY)

    return {
        "status": "ready" if groq_configured else "not_ready",
        "groq_configured": groq_configured,
        "model": GROQ_MODEL,
        "redis": redis_status,
        "note": (
            "redis='unreachable' or 'not_configured' does not block "
            "readiness - the service fails open to an in-memory-only "
            "validation cache, but consistency then only holds within "
            "a single process. See REDIS_URL in .env."
        ),
    }


@app.get("/api/v1/cache/stats")
def cache_stats() -> dict[str, Any]:
    with _validation_cache_lock:
        local_size = len(_validation_cache)
        local_canonical_size = len(_canonical_cache)

    redis_backed = False
    redis_size: int | None = None
    redis_canonical_size: int | None = None
    if _redis_client is not None:
        try:
            redis_backed = True
            redis_size = 0
            cursor = 0
            while True:
                cursor, keys = _redis_client.scan(
                    cursor=cursor,
                    match=CACHE_KEY_PREFIX + "*",
                    count=500,
                )
                redis_size += len(keys)
                if cursor == 0:
                    break

            redis_canonical_size = 0
            cursor = 0
            while True:
                cursor, keys = _redis_client.scan(
                    cursor=cursor,
                    match=CANONICAL_KEY_PREFIX + "*",
                    count=500,
                )
                redis_canonical_size += len(keys)
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("Redis SCAN failed for cache_stats: %s", exc)
            redis_backed = False
            redis_size = None
            redis_canonical_size = None

    return {
        "redis_backed": redis_backed,
        "redis_cached_results": redis_size,
        "in_memory_cached_results": local_size,
        "redis_canonical_estimates": redis_canonical_size,
        "in_memory_canonical_estimates": local_canonical_size,
    }


@app.post("/api/v1/cache/clear")
def cache_clear() -> dict[str, str]:
    """Force every task to be re-validated from scratch on its next call.

    Not needed for normal operation - an unchanged task is always
    consistent by design, in both the Redis-backed and in-memory-only
    cases. Provided for testing and for the rare case where a fresh
    LLM opinion is deliberately wanted despite no field having changed.
    Clears both the Redis-backed cache (if configured) and this
    process's local in-memory copy.
    """
    clear_validation_cache()
    return {"status": "cache cleared"}


@app.post(
    "/api/v1/task/validate",
    response_model=SingleTaskResponse,
)
def validate_single_task(
    task: TaskInput,
) -> SingleTaskResponse:
    result = validate_with_groq(task)

    return SingleTaskResponse(
        status=(
            "FAILED"
            if result.decision == "ERROR"
            else "COMPLETED"
        ),
        result=result,
    )


@app.post(
    "/api/v1/backlog/validate",
    response_model=BulkResponse,
)
def validate_backlog_json(
    request: BulkTaskValidationRequest,
) -> BulkResponse:
    if not request.tasks:
        raise HTTPException(
            status_code=400,
            detail="At least one task is required.",
        )

    results = validate_tasks_concurrently(
        request.tasks,
        project=request.project,
        sprint=request.sprint,
    )

    return build_bulk_response(results)


@app.post(
    "/api/v1/backlog/upload-and-validate",
    response_model=BulkResponse,
)
async def upload_and_validate(
    file: UploadFile = File(...),
) -> BulkResponse:
    """Optional Excel endpoint retained for manual testing."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing",
        )

    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(
            status_code=400,
            detail="Only .xlsx files are supported",
        )

    try:
        file_bytes = await file.read()

        dataframe = pd.read_excel(
            io.BytesIO(file_bytes),
            engine="openpyxl",
        )

        tasks = dataframe_to_tasks(dataframe)

        if not tasks:
            raise HTTPException(
                status_code=400,
                detail="No valid tasks found in workbook",
            )

        results = validate_tasks_concurrently(tasks)

        return build_bulk_response(results)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {error}",
        ) from error