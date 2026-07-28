from __future__ import annotations

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
- "suggested_estimated_hours": return 0. This is a reserved sentinel value meaning "not applicable / not validated," and must NOT be read as an approved or recommended estimate. Do not return the original estimated_hours value here, since doing so could be misread as confirming the original estimate is acceptable.
- "recommendation": explicitly state that the estimate could not be validated because the description is missing or insufficient, and that the user must provide a meaningful task description before re-validation.

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

suggested_estimated_hours = 0 (reserved sentinel meaning "not validated" — see MANDATORY TASK DESCRIPTION section)

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

"The estimate could not be validated because the task description is missing or insufficient. Please provide a meaningful description and resubmit for validation."

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

always return decision = CANNOT_VALIDATE_ESTIMATE with suggested_estimated_hours = 0, per the MANDATORY TASK DESCRIPTION section.

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
        disclaimer = (
            "This is an AI-generated estimation review based only on "
            "the supplied task information; human review is required before applying any changes."
        )
        if disclaimer.lower() not in recommendation.lower():
            recommendation = f"{recommendation} {disclaimer}".strip()
        parsed["recommendation"] = recommendation

        return ValidationResult.model_validate(parsed)

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