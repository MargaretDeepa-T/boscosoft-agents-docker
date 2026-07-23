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
You are Agent 1, the Task and Estimation Validation Agent
for Boscosoft's Agile Project Management Tool.

Validate the task title, task description, scope, complexity,
and estimated hours using only the information provided.

Decision rules:
- PROCEED: the title, description, scope, complexity, and estimate are reasonable.
- REVIEW_ESTIMATE: the title and description are acceptable, but the estimate is likely high or low.
- REWRITE_TASK: the title or description needs correction, but the estimate remains reasonable.
- REWRITE_AND_REESTIMATE: the task definition and estimate both need correction.
- CANNOT_VALIDATE_ESTIMATE: the description is too incomplete to judge the estimate reliably.

Mandatory output rules:
- Return valid JSON only.
- Never return null or an empty value for any suggested field.
- suggested_task_title must contain the exact original title when it is acceptable;
  otherwise, return a corrected, clear, specific, action-oriented title.
- suggested_task_description must contain the exact original description when it is acceptable;
  otherwise, return a corrected description that explains scope, expected outcome,
  major activities, and functional boundary without inventing unsupported requirements.
- For PROCEED, REWRITE_TASK, or CANNOT_VALIDATE_ESTIMATE,
  suggested_estimated_hours must equal the original estimated hours.
- For REVIEW_ESTIMATE or REWRITE_AND_REESTIMATE,
  suggested_estimated_hours must be a realistic revised number and MUST be different
  from the original estimated hours.
- Do not use text such as "No changes required" in suggested fields.
- Use cautious wording such as appears reasonable, may be low, or may be high.
- Human review is always required before applying changes.
The estimate is an AI-generated review recommendation based only on the
provided task information. It is not an authoritative project estimate.
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
  "recommendation": "A concise recommendation for human review, clearly stating that the estimate is AI-generated and requires human approval"
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
            "the supplied task information; human approval is required."
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