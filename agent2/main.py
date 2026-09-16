"""
Agent 2 - Timesheet Execution Review & Manager Summary Agent
==============================================================

This FastAPI application reviews an employee's submitted timesheet entries
against ONE approved backlog task within a selected date range.

The Agent assists the Project Manager in validating whether recorded work
genuinely supports the assigned backlog task. It performs validation and
returns recommendations only - it never modifies any data, and all numeric
calculations are performed locally in Python (never by the LLM).
"""

# ==============================================================
# 1. IMPORTS
# ==============================================================

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# ==============================================================
# 2. ENVIRONMENT VARIABLE LOADING
# ==============================================================

load_dotenv()

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
GROQ_MODEL: Optional[str] = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing required environment variable: GROQ_API_KEY. "
        "Please set it in the .env file."
    )

if not GROQ_MODEL:
    raise RuntimeError(
        "Missing required environment variable: GROQ_MODEL. "
        "Please set it in the .env file."
    )

# ==============================================================
# 3. CONFIGURATION
# ==============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("agent2")

MAX_LLM_RETRIES = int(os.getenv("MAX_LLM_RETRIES", "3"))
GROQ_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("GROQ_REQUEST_TIMEOUT_SECONDS", "60")
)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "1.5"))
GROQ_MAX_COMPLETION_TOKENS = int(
    os.getenv("GROQ_MAX_COMPLETION_TOKENS", "1800")
)

# Client-side ceiling on concurrent Groq calls from this service. Each
# review is a single Groq call, but under concurrent manager traffic this
# is what keeps a burst of simultaneous reviews from tripping the
# account's free-tier RPM (requests-per-minute) limit.
GROQ_MAX_CONCURRENT_REQUESTS = max(
    1,
    int(os.getenv("GROQ_MAX_CONCURRENT_REQUESTS", "3")),
)

# Long free-text fields cost tokens without adding review value beyond a
# point; truncating keeps one oversized entry from eating a
# disproportionate share of the per-minute token budget.
MAX_TEXT_FIELD_CHARS = int(os.getenv("MAX_TEXT_FIELD_CHARS", "2000"))

# Upper bound on how many backlog tasks may be submitted in a single bulk
# review request. Keeps one request from creating unbounded server work.
MAX_TASKS_PER_BULK_REQUEST = max(
    1,
    int(os.getenv("MAX_TASKS_PER_BULK_REQUEST", "20")),
)

# Upper bound on the number of tasks processed concurrently within one
# bulk request. This is independent of, and layered on top of, the Groq
# concurrency gate below - each worker thread still has to acquire that
# gate before it can actually call Groq.
BULK_MAX_WORKERS = max(
    1,
    int(os.getenv("BULK_MAX_WORKERS", "3")),
)

# In-memory cache of completed reviews, keyed on a hash of everything
# that can affect the outcome (employee, backlog task, timesheet
# entries, report period). The LLM is not perfectly deterministic even
# at low temperature, so without this, re-submitting the exact same,
# unchanged task could yield a different decision/wording each time.
# With it, an identical resubmission returns the exact same result
# without calling the LLM again.
REVIEW_CACHE_ENABLED = os.getenv("REVIEW_CACHE_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)
REVIEW_CACHE_TTL_SECONDS = int(os.getenv("REVIEW_CACHE_TTL_SECONDS", "86400"))
REVIEW_CACHE_MAX_ENTRIES = max(
    1,
    int(os.getenv("REVIEW_CACHE_MAX_ENTRIES", "1000")),
)

_groq_concurrency_gate = threading.Semaphore(GROQ_MAX_CONCURRENT_REQUESTS)

# Fields the LLM must never emit. The Agent only ever issues a
# recommendation - the Project Manager makes the actual approval decision -
# so any workflow/approval-style key slipping into the LLM output is a
# prompt-compliance violation. It is logged and stripped, not trusted.
FORBIDDEN_LLM_OUTPUT_KEYS = {
    "approval_status",
    "approved",
    "rejected",
    "ready_for_approval",
    "status",
}

# ==============================================================
# 4. FASTAPI APPLICATION
# ==============================================================

# app = FastAPI(
#     title="Agent 2 - Timesheet Execution Review & Manager Summary Agent",
#     description=(
#         "Reviews an employee's submitted timesheet entries against ONE "
#         "approved backlog task within a selected date range, and returns "
#         "validation results and recommendations for the Project Manager."
#     ),
#     version="1.0.0",
# )
app = FastAPI(
    title="Agent 2 - Timesheet Execution Review & Manager Summary Agent",
    description=(
        "Reviews an employee's submitted timesheet entries against ONE "
        "approved backlog task within a selected date range, and returns "
        "validation results and recommendations for the Project Manager."
    ),
    root_path="/agent2"
)

# ==============================================================
# 5. ENUMS
# ==============================================================


class ReportType(str, Enum):
    """Supported reporting periods for a timesheet review."""

    EOD = "EOD"
    EOW = "EOW"
    EOM = "EOM"
    DATE_RANGE = "DATE_RANGE"


class Decision(str, Enum):
    """Final decision returned by the Agent for the reviewed task."""

    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    CORRECTION_REQUIRED = "CORRECTION_REQUIRED"


# ==============================================================
# 6. REQUEST MODELS
# ==============================================================


class Employee(BaseModel):
    """Employee whose timesheet entries are being reviewed."""

    employee_id: str
    employee_name: str
    designation: Optional[str] = None


class BacklogTask(BaseModel):
    """The single approved backlog task the timesheet is reviewed against."""

    task_id: str
    project: str
    sprint: str
    module: str
    feature: str
    task_title: str
    task_description: str
    task_type: Optional[str] = None
    complexity: str
    estimated_hours: float = Field(..., gt=0)
    scheduled_hours: float = Field(..., gt=0)
    task_status: str


class TimesheetEntry(BaseModel):
    """A single timesheet entry recorded against the backlog task."""

    entry_id: str
    date: date
    discipline: str
    activity_description: str
    planned_hours: float = Field(..., ge=0)
    actual_hours: float = Field(..., ge=0)


class TimesheetReviewRequest(BaseModel):
    """Request payload for the manager timesheet review API."""

    report_type: ReportType
    start_date: date
    end_date: date
    employee: Employee
    backlog_task: BacklogTask
    timesheet_entries: List[TimesheetEntry]

    @model_validator(mode="after")
    def validate_request(self) -> "TimesheetReviewRequest":
        """Ensure the date range and timesheet entries are sane."""
        if self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date.")
        if len(self.timesheet_entries) == 0:
            raise ValueError("timesheet_entries cannot be empty.")
        return self


class TaskTimesheetGroup(BaseModel):
    """One backlog task paired with the timesheet entries logged against it.

    Used only inside a bulk review request - one employee may submit
    several of these, one per assigned backlog task.
    """

    backlog_task: BacklogTask
    timesheet_entries: List[TimesheetEntry]

    @model_validator(mode="after")
    def validate_task_group(self) -> "TaskTimesheetGroup":
        """Every task in a bulk request must have at least one entry."""
        if not self.timesheet_entries:
            raise ValueError("timesheet_entries cannot be empty.")
        return self


class BulkTimesheetReviewRequest(BaseModel):
    """Request payload for reviewing multiple backlog tasks for one employee.

    Each task is validated completely independently of the others - the
    grouping here only exists to describe the batch; it is never sent to
    the LLM as a single combined prompt.
    """

    report_type: ReportType
    start_date: date
    end_date: date
    employee: Employee
    tasks: List[TaskTimesheetGroup]

    @model_validator(mode="after")
    def validate_request(self) -> "BulkTimesheetReviewRequest":
        """Validate the date range, task list size, and cross-task uniqueness."""
        if self.end_date < self.start_date:
            raise ValueError("end_date cannot be before start_date.")

        if not self.tasks:
            raise ValueError("tasks cannot be empty.")

        if len(self.tasks) > MAX_TASKS_PER_BULK_REQUEST:
            raise ValueError(
                "Bulk request contains "
                f"{len(self.tasks)} tasks, which exceeds the configured "
                f"maximum of {MAX_TASKS_PER_BULK_REQUEST} "
                "(MAX_TASKS_PER_BULK_REQUEST)."
            )

        seen_task_ids: set[str] = set()
        seen_entry_ids: set[str] = set()

        for task_group in self.tasks:
            task_id = task_group.backlog_task.task_id
            if task_id in seen_task_ids:
                raise ValueError(
                    f"Duplicate task_id found in bulk request: {task_id}"
                )
            seen_task_ids.add(task_id)

            for entry in task_group.timesheet_entries:
                if entry.entry_id in seen_entry_ids:
                    raise ValueError(
                        "Duplicate entry_id found across bulk request: "
                        f"{entry.entry_id}"
                    )
                seen_entry_ids.add(entry.entry_id)

                if entry.date < self.start_date or entry.date > self.end_date:
                    raise ValueError(
                        f"Timesheet entry {entry.entry_id} has date "
                        f"{entry.date.isoformat()}, which falls outside the "
                        f"review period {self.start_date.isoformat()} - "
                        f"{self.end_date.isoformat()}."
                    )

        return self


# ==============================================================
# 7. RESPONSE MODELS
# ==============================================================


class ReviewPeriod(BaseModel):
    start_date: date
    end_date: date


class ValidationResults(BaseModel):
    task_assignment_valid: bool
    activity_alignment_valid: bool
    discipline_valid: bool
    effort_valid: bool
    completion_valid: bool
    duplicate_activity_found: bool
    irrelevant_activity_found: bool


class Coverage(BaseModel):
    analysis: bool
    design: bool
    development: bool
    testing: bool
    documentation: bool


class EffortSummary(BaseModel):
    estimated_hours: float
    scheduled_hours: float
    total_planned_hours: float
    total_actual_hours: float
    remaining_hours: float
    variance_hours: float
    completion_percentage: float


class TimesheetReviewResponse(BaseModel):
    """Final response schema returned by the manager timesheet review API."""

    status: str
    decision: Decision
    employee_id: str
    employee_name: str
    task_id: str
    review_period: ReviewPeriod
    validation_results: ValidationResults
    coverage: Coverage
    effort_summary: EffortSummary
    strengths: List[str]
    issues: List[str]
    manager_actions: List[str]


class LLMReviewResult(BaseModel):
    """Strict schema for the LLM's qualitative reasoning output only.

    Validated immediately after the raw JSON is parsed, before it is
    merged with the locally computed effort figures. Extra keys (e.g. a
    stray ``approval_status`` the LLM must never emit, per business rule)
    are ignored here rather than failing the whole request - callers check
    for them separately via FORBIDDEN_LLM_OUTPUT_KEYS and log a warning.
    """

    model_config = ConfigDict(extra="ignore")

    decision: Decision
    validation_results: ValidationResults
    coverage: Coverage
    strengths: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    manager_actions: List[str] = Field(default_factory=list)


class TaskReviewFailure(BaseModel):
    """Safe, client-facing record of one task that failed review in a bulk request."""

    task_id: str
    error_type: str
    error_message: str


class BulkTimesheetReviewResponse(BaseModel):
    """Final response schema returned by the bulk manager timesheet review API."""

    status: str
    employee_id: str
    employee_name: str
    report_type: ReportType
    review_period: ReviewPeriod
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    accepted_tasks: int
    review_tasks: int
    correction_required_tasks: int
    total_planned_hours: float
    total_actual_hours: float
    results: List[TimesheetReviewResponse]
    failures: List[TaskReviewFailure]


class TaskReviewError(Exception):
    """A single task's review could not be completed.

    Carries a safe, client-facing error_type/message pair (no credentials,
    stack traces, or other internal detail) plus the HTTP status code the
    single-task endpoint should surface if this bubbles up there. The full
    technical detail is always logged separately, server-side only, at the
    point this is raised.
    """

    def __init__(self, error_type: str, message: str, status_code: int = 502) -> None:
        self.error_type = error_type
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ==============================================================
# 7b. REVIEW CACHE
# ==============================================================
#
# Purpose: the LLM is not perfectly deterministic even at low temperature,
# so re-validating the exact same, unchanged task twice can otherwise
# produce a different decision or differently-worded issues each time.
# This cache makes repeat validation of an unchanged task idempotent -
# the same input always returns the same previously-computed result,
# without a new LLM call - while still letting a genuinely changed task
# (different hours, different activity text, etc.) get a fresh review.


class _ReviewCache:
    """Thread-safe, bounded, TTL in-memory cache of completed task reviews.

    Keyed on a canonical hash of everything that can affect the review
    outcome. Bounded by REVIEW_CACHE_MAX_ENTRIES with simple LRU eviction,
    and entries expire after REVIEW_CACHE_TTL_SECONDS so a cache is never
    served indefinitely.

    In-memory only: cache contents are per-process and are lost on
    restart. For multi-instance deployments, back this with Redis instead
    (same get/set/clear interface) so all instances share one cache.
    """

    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, Tuple[float, TimesheetReviewResponse]]" = (
            OrderedDict()
        )

    def get(self, key: str) -> Optional["TimesheetReviewResponse"]:
        if self._ttl_seconds <= 0:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            cached_at, response = entry
            if (time.time() - cached_at) > self._ttl_seconds:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return response

    def set(self, key: str, response: "TimesheetReviewResponse") -> None:
        with self._lock:
            self._store[key] = (time.time(), response)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> int:
        """Remove all cached entries. Returns the number removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": REVIEW_CACHE_ENABLED,
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl_seconds,
            }


_review_cache = _ReviewCache(
    max_entries=REVIEW_CACHE_MAX_ENTRIES, ttl_seconds=REVIEW_CACHE_TTL_SECONDS
)


def compute_review_cache_key(request: "TimesheetReviewRequest") -> str:
    """Build a canonical, order-independent hash of everything that can
    affect a task's review outcome.

    Timesheet entries are sorted by entry_id before hashing so that the
    same set of entries in a different order still produces the same key.
    Using model_dump(mode="json") (rather than str()) keeps the hash
    stable and type-safe across dates, enums, and floats.
    """
    canonical = {
        "employee": request.employee.model_dump(mode="json"),
        "backlog_task": request.backlog_task.model_dump(mode="json"),
        "timesheet_entries": sorted(
            (entry.model_dump(mode="json") for entry in request.timesheet_entries),
            key=lambda entry: entry["entry_id"],
        ),
        "report_type": request.report_type.value,
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
    }
    canonical_json = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ==============================================================
# 8. LOCAL CALCULATIONS
# ==============================================================


def calculate_effort_summary(
    backlog_task: BacklogTask, timesheet_entries: List[TimesheetEntry]
) -> EffortSummary:
    """Compute all numeric effort figures locally.

    The LLM must never calculate numbers - all arithmetic is performed
    here in Python to guarantee correctness.
    """
    total_planned_hours = sum(entry.planned_hours for entry in timesheet_entries)
    total_actual_hours = sum(entry.actual_hours for entry in timesheet_entries)

    remaining_hours = backlog_task.estimated_hours - total_actual_hours
    if remaining_hours < 0:
        remaining_hours = 0.0

    variance_hours = total_actual_hours - total_planned_hours

    if backlog_task.estimated_hours > 0:
        completion_percentage = round(
            (total_actual_hours / backlog_task.estimated_hours) * 100, 2
        )
    else:
        completion_percentage = 0.0

    return EffortSummary(
        estimated_hours=backlog_task.estimated_hours,
        scheduled_hours=backlog_task.scheduled_hours,
        total_planned_hours=round(total_planned_hours, 2),
        total_actual_hours=round(total_actual_hours, 2),
        remaining_hours=round(remaining_hours, 2),
        variance_hours=round(variance_hours, 2),
        completion_percentage=completion_percentage,
    )


# ==============================================================
# 8b. PROMPT-SIZE GUARD
# ==============================================================


def truncate_for_prompt(text: str, max_chars: int = MAX_TEXT_FIELD_CHARS) -> str:
    """Cap long free-text fields before they go into a Groq prompt.

    Only affects what is sent to the model - the values stored and
    returned in the response are always the untouched originals.
    """
    if not text or len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + " …[truncated for length]"


# ==============================================================
# 9. PROMPT BUILDER
# ==============================================================


def build_prompt(request: TimesheetReviewRequest, effort_summary: EffortSummary) -> str:
    """Build the reasoning prompt sent to the LLM.

    The prompt supplies the backlog task, the timesheet entries, and the
    pre-computed effort figures, and instructs the LLM to perform ONLY
    qualitative reasoning (never numeric calculation) and to return ONLY
    the final JSON object matching the required schema.
    """
    timesheet_entries_json = json.dumps(
        [
            {
                "entry_id": entry.entry_id,
                "date": entry.date.isoformat(),
                "discipline": entry.discipline,
                "activity_description": truncate_for_prompt(
                    entry.activity_description
                ),
                "planned_hours": entry.planned_hours,
                "actual_hours": entry.actual_hours,
            }
            for entry in request.timesheet_entries
        ],
        indent=2,
    )

    backlog_task = request.backlog_task
    prompt_task_description = truncate_for_prompt(
        backlog_task.task_description
    )

    prompt = f"""You are an expert Project Management Assistant AI. Your job is to
review an employee's timesheet entries against ONE approved backlog task and
determine whether the recorded work genuinely supports that task.

You DO NOT modify any data. You ONLY validate and reason. You MUST NEVER
calculate numeric totals, hours, or percentages - all numeric figures have
already been computed by the system and are provided to you below for
reference only.

============================================================
CRITICAL BUSINESS RULE - READ CAREFULLY
============================================================
This Agent NEVER approves or rejects a timesheet. You are an AI reviewer
only. The Project Manager makes the final approval decision, not you.

The field "decision" you return is ONLY the Agent's recommendation to the
Project Manager. Its only allowed values are ACCEPT, REVIEW, or
CORRECTION_REQUIRED.

You MUST NEVER return an "approval_status" field, or values such as
"approved", "rejected", "ready_for_approval", or any other workflow /
approval status. Output ONLY the fields defined in the schema below.

============================================================
BACKLOG TASK
============================================================
Task ID: {backlog_task.task_id}
Project: {backlog_task.project}
Sprint: {backlog_task.sprint}
Module: {backlog_task.module}
Feature: {backlog_task.feature}
Task Title: {backlog_task.task_title}
Task Description: {prompt_task_description}
Task Type: {backlog_task.task_type or "N/A"}
Complexity: {backlog_task.complexity}
Task Status: {backlog_task.task_status}

============================================================
EMPLOYEE
============================================================
Employee ID: {request.employee.employee_id}
Employee Name: {request.employee.employee_name}
Designation: {request.employee.designation or "N/A"}

============================================================
REVIEW PERIOD
============================================================
Report Type: {request.report_type.value}
Start Date: {request.start_date.isoformat()}
End Date: {request.end_date.isoformat()}

============================================================
TIMESHEET ENTRIES
============================================================
{timesheet_entries_json}

============================================================
PRE-COMPUTED EFFORT FIGURES (reference only - do not recompute)
============================================================
Estimated Hours: {effort_summary.estimated_hours}
Scheduled Hours: {effort_summary.scheduled_hours}
Total Planned Hours: {effort_summary.total_planned_hours}
Total Actual Hours: {effort_summary.total_actual_hours}
Remaining Hours: {effort_summary.remaining_hours}
Variance Hours: {effort_summary.variance_hours}
Completion Percentage: {effort_summary.completion_percentage}

============================================================
REASONING PROCESS - FOLLOW THIS EXACT ORDER (internal - do not output these steps)
============================================================
Step 1: Understand the assigned backlog task - its title, description,
        module, feature, complexity, and current task status.
Step 2: Review every timesheet activity submitted for the review period.
Step 3: Determine whether every activity genuinely belongs to the
        assigned backlog task (task assignment + activity alignment).
Step 4: Validate whether the discipline selected for each activity is
        appropriate for the work described.
Step 5: Determine coverage of the following work phases: Analysis,
        Design, Development, Testing, Documentation.
Step 6: Determine whether enough evidence exists across the entries to
        justify the task's current status ({backlog_task.task_status}).
Step 7: Identify duplicate work entries (repeated or near-identical
        activities).
Step 8: Identify irrelevant work (activities unrelated to the assigned
        task).
Step 9: Generate strengths - concrete, positive observations.
Step 10: Generate issues - missing, incorrect, or unsupported work.
Step 11: Generate manager actions - concrete next steps for the Project
        Manager to take before making their own approval decision.
Step 12: Generate the final recommendation (the "decision" field) using
        the definitions below.

============================================================
DECISION DEFINITIONS (choose exactly one)
============================================================
ACCEPT
  The timesheet aligns well with the assigned backlog task. No
  significant issues exist. Minor observations are acceptable.

REVIEW
  The work generally supports the task. However, one or more
  observations require the Project Manager's attention before taking
  action.

CORRECTION_REQUIRED
  The employee should correct the submitted timesheet. Examples include:
  activities unrelated to the assigned task, incorrect discipline,
  unsupported completion, duplicate work, or insufficient work evidence.

============================================================
OUTPUT RULES
============================================================
- Return ONLY a single valid JSON object. No markdown, no explanations, no comments.
- Never return null for any field. Use empty arrays ([]) when there is nothing to report.
- Do NOT include "approval_status", "approved", "rejected",
  "ready_for_approval", "status", or any field not listed below.
- The JSON object MUST exactly match this schema (keys and types) and no other fields:

{{
  "decision": "ACCEPT | REVIEW | CORRECTION_REQUIRED",
  "validation_results": {{
    "task_assignment_valid": true,
    "activity_alignment_valid": true,
    "discipline_valid": true,
    "effort_valid": true,
    "completion_valid": true,
    "duplicate_activity_found": false,
    "irrelevant_activity_found": false
  }},
  "coverage": {{
    "analysis": true,
    "design": true,
    "development": true,
    "testing": true,
    "documentation": true
  }},
  "strengths": ["string", "..."],
  "issues": ["string", "..."],
  "manager_actions": ["string", "..."]
}}

Return ONLY the JSON object described above."""

    return prompt


# ==============================================================
# 10. GROQ CLIENT
# ==============================================================


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

    This is a 400 from Groq's JSON-mode validator when the model's
    generated text failed to parse as JSON. It is a sampling hiccup, not
    a malformed request - the same prompt commonly succeeds on retry -
    so it is treated as transient rather than permanent, unlike other
    4xx errors.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = body.get("code") or body.get("error", {}).get("code")
        if code == "json_validate_failed":
            return True
    return "json_validate_failed" in str(exc)


groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_REQUEST_TIMEOUT_SECONDS)


def call_groq_llm(prompt: str) -> str:
    """Call the Groq chat completion API and return the raw text response.

    Bounded by GROQ_MAX_CONCURRENT_REQUESTS so a burst of simultaneous
    reviews never has more requests in flight against Groq than the
    account's free-tier RPM budget can absorb.

    Raises:
        GroqAuthenticationFailure: invalid/missing credentials.
        GroqRateLimitFailure: HTTP 429 - caller should back off and retry.
        GroqTransientFailure: timeout, connection, or 5xx error - the
            caller may safely retry.
        GroqPermanentFailure: any other non-retryable API error.
    """
    with _groq_concurrency_gate:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise reasoning engine that returns "
                            "only valid JSON with no additional text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content or ""

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
                logger.warning("Groq API returned a server error: %s", exc)
                raise GroqTransientFailure(str(exc)) from exc
            if exc.status_code == 400 and _is_json_validate_failed(exc):
                logger.warning(
                    "Groq JSON validation failed (400 json_validate_failed), "
                    "treating as retryable: %s",
                    exc,
                )
                raise GroqTransientFailure(str(exc)) from exc
            logger.error("Groq API returned a client error: %s", exc)
            raise GroqPermanentFailure(str(exc)) from exc


# ==============================================================
# 11. JSON PARSER
# ==============================================================


def parse_llm_json(raw_text: str) -> dict:
    """Parse the raw LLM text response into a JSON object.

    Strips common markdown code-fence wrapping that some models add
    even when instructed not to, then parses with the standard json
    module.

    Raises:
        ValueError: if the text cannot be parsed as valid JSON.
    """
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ==============================================================
# 12. RETRY LOGIC
# ==============================================================


def get_llm_review(prompt: str) -> LLMReviewResult:
    """Call the LLM and return its validated qualitative review.

    Retries up to MAX_LLM_RETRIES times, with a short backoff between
    attempts, on transient Groq failures, invalid JSON, or schema
    validation failures. Authentication failures and other non-retryable
    API errors fail immediately without retrying.

    Raises:
        TaskReviewError: with a safe, client-facing error_type/message and
        the HTTP status code the caller should use if this is the only
        task being reviewed. Full technical detail is always logged here,
        server-side only, before the safe error is raised.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            raw_response = call_groq_llm(prompt)
            parsed = parse_llm_json(raw_response)

            stray_keys = FORBIDDEN_LLM_OUTPUT_KEYS & parsed.keys()
            if stray_keys:
                logger.warning(
                    "LLM response included forbidden workflow field(s) %s; "
                    "ignoring them - decision is a recommendation only.",
                    sorted(stray_keys),
                )

            return LLMReviewResult.model_validate(parsed)

        except GroqAuthenticationFailure as exc:
            logger.error("Authentication with the AI service failed: %s", exc)
            raise TaskReviewError(
                error_type="AuthenticationFailure",
                message=(
                    "The task could not be reviewed because authentication "
                    "with the AI service failed. Please contact the system "
                    "administrator."
                ),
                status_code=401,
            ) from exc

        except GroqPermanentFailure as exc:
            logger.error("The AI service returned a non-retryable error: %s", exc)
            raise TaskReviewError(
                error_type="AIServiceFailure",
                message=(
                    "The task could not be reviewed because the AI service "
                    "returned an error."
                ),
                status_code=502,
            ) from exc

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

        except (GroqTransientFailure, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "LLM attempt %d/%d failed (%s): %s",
                attempt,
                MAX_LLM_RETRIES,
                type(exc).__name__,
                exc,
            )
            if attempt < MAX_LLM_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error(
        "LLM failed to produce a valid review after %d attempts. Last error (%s): %s",
        MAX_LLM_RETRIES,
        type(last_error).__name__ if last_error else "unknown",
        last_error,
    )

    if isinstance(last_error, GroqRateLimitFailure):
        raise TaskReviewError(
            error_type="GroqRateLimitFailure",
            message=(
                "The task could not be reviewed because the AI service was "
                "temporarily unavailable."
            ),
            status_code=502,
        ) from last_error

    if isinstance(last_error, (json.JSONDecodeError, ValidationError)):
        raise TaskReviewError(
            error_type="InvalidAIResponse",
            message=(
                "The task could not be reviewed because the AI service "
                "returned a response that could not be processed."
            ),
            status_code=502,
        ) from last_error

    raise TaskReviewError(
        error_type="AIServiceTransientFailure",
        message=(
            "The task could not be reviewed because of a temporary AI "
            "service issue. Please retry."
        ),
        status_code=502,
    ) from last_error


# ==============================================================
# 13. RESPONSE VALIDATION / ASSEMBLY
# ==============================================================


def assemble_response(
    request: TimesheetReviewRequest,
    llm_result: LLMReviewResult,
    effort_summary: EffortSummary,
) -> TimesheetReviewResponse:
    """Combine the LLM's validated reasoning with the locally computed figures.

    Raises:
        TaskReviewError: 502, if the merged payload fails schema validation.
    """
    try:
        payload = {
            "status": "COMPLETED",
            "decision": llm_result.decision,
            "employee_id": request.employee.employee_id,
            "employee_name": request.employee.employee_name,
            "task_id": request.backlog_task.task_id,
            "review_period": {
                "start_date": request.start_date,
                "end_date": request.end_date,
            },
            "validation_results": llm_result.validation_results,
            "coverage": llm_result.coverage,
            "effort_summary": effort_summary,
            "strengths": llm_result.strengths,
            "issues": llm_result.issues,
            "manager_actions": llm_result.manager_actions,
        }
        return TimesheetReviewResponse.model_validate(payload)

    except ValidationError as exc:
        logger.error("Merged review payload failed schema validation: %s", exc)
        raise TaskReviewError(
            error_type="ResponseValidationFailure",
            message=(
                "The task could not be reviewed because the AI-generated "
                "review failed validation."
            ),
            status_code=502,
        ) from exc


# ==============================================================
# 13b. SHARED SINGLE-TASK REVIEW PIPELINE
# ==============================================================


def process_single_task_review(
    request: TimesheetReviewRequest,
) -> TimesheetReviewResponse:
    """Run the complete review pipeline for exactly one backlog task.

    This is the single reusable core used by both the single-task endpoint
    and (once per task, independently) by the bulk endpoint:

        0. Check the review cache for an identical, previously-reviewed
           (employee, task, entries, period) combination.
        1. Calculate the effort summary locally in Python.
        2. Build the LLM prompt for this task only.
        3. Call the LLM and validate its qualitative response.
        4. Assemble and validate the final response.
        5. Store the result in the cache for future identical requests.

    Raises:
        TaskReviewError: on any failure in steps 2-4, with a safe,
        client-facing error_type/message and a suggested HTTP status code.
        Full technical detail is always logged at the point of failure.
    """
    cache_key: Optional[str] = None
    if REVIEW_CACHE_ENABLED:
        cache_key = compute_review_cache_key(request)
        cached_response = _review_cache.get(cache_key)
        if cached_response is not None:
            logger.info(
                "Review cache hit | task_id=%s cache_key=%s",
                request.backlog_task.task_id,
                cache_key[:12],
            )
            return cached_response

    effort_summary = calculate_effort_summary(
        request.backlog_task, request.timesheet_entries
    )
    prompt = build_prompt(request, effort_summary)
    llm_result = get_llm_review(prompt)
    response = assemble_response(request, llm_result, effort_summary)

    if cache_key:
        _review_cache.set(cache_key, response)

    return response


def build_task_review_request(
    bulk_request: "BulkTimesheetReviewRequest",
    task_group: TaskTimesheetGroup,
) -> TimesheetReviewRequest:
    """Convert one task group from a bulk request into a standalone
    TimesheetReviewRequest, reusing the shared report-level fields."""
    return TimesheetReviewRequest(
        report_type=bulk_request.report_type,
        start_date=bulk_request.start_date,
        end_date=bulk_request.end_date,
        employee=bulk_request.employee,
        backlog_task=task_group.backlog_task,
        timesheet_entries=task_group.timesheet_entries,
    )


# ==============================================================
# 14. ERROR HANDLING (global exception handlers)
# ==============================================================


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected exceptions."""
    logger.exception(
        "Unexpected error while handling request %s: %s", request.url.path, exc
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"},
    )


# ==============================================================
# 15. HEALTH API
# ==============================================================


@app.get("/health", tags=["Health"], summary="Health check")
def health() -> dict:
    """Return the health status of the service."""
    return {
        "status": "UP",
        "service": "Agent 2",
        "version": "1.0.0",
    }


# ==============================================================
# 15b. REVIEW CACHE ADMIN API
# ==============================================================


@app.get(
    "/api/v1/admin/cache/stats",
    tags=["Admin"],
    summary="View review cache statistics",
)
def get_cache_stats() -> dict:
    """Return the current size and configuration of the review cache."""
    return _review_cache.stats()


@app.post(
    "/api/v1/admin/cache/clear",
    tags=["Admin"],
    summary="Clear the review cache",
)
def clear_cache() -> dict:
    """Clear all cached reviews.

    Use this after a prompt or model change, or whenever a fresh review
    is needed for tasks that were previously cached.
    """
    removed = _review_cache.clear()
    logger.info("Review cache cleared | entries_removed=%d", removed)
    return {"status": "CLEARED", "entries_removed": removed}


# ==============================================================
# 16. MANAGER REVIEW API
# ==============================================================


@app.post(
    "/api/v1/manager/timesheet-review",
    response_model=TimesheetReviewResponse,
    tags=["Manager Review"],
    summary="Review employee timesheet entries against a backlog task",
)
def review_timesheet(request: TimesheetReviewRequest) -> TimesheetReviewResponse:
    """Review an employee's timesheet entries against one backlog task.

    Performs local numeric calculations in Python, delegates qualitative
    validation and reasoning to the LLM (with retry logic for transient
    failures and invalid/non-conforming JSON), validates the merged result
    against the response schema, and returns the Agent's recommendation
    for the Project Manager. The Agent never approves or rejects work
    itself - that decision always belongs to the Project Manager.
    """
    review_id = uuid4().hex[:8]
    started_at = time.perf_counter()
    logger.info(
        "[%s] Starting review | employee=%s task=%s entries=%d",
        review_id,
        request.employee.employee_id,
        request.backlog_task.task_id,
        len(request.timesheet_entries),
    )

    try:
        response = process_single_task_review(request)
    except TaskReviewError as exc:
        logger.error(
            "[%s] Review failed | task=%s error_type=%s",
            review_id,
            request.backlog_task.task_id,
            exc.error_type,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    logger.info(
        "[%s] Completed review | decision=%s elapsed_ms=%s",
        review_id,
        response.decision.value,
        elapsed_ms,
    )
    return response


# ==============================================================
# 16b. BULK MANAGER REVIEW API
# ==============================================================


def _process_one_bulk_task(
    bulk_id: str,
    index: int,
    bulk_request: BulkTimesheetReviewRequest,
    task_group: TaskTimesheetGroup,
) -> Tuple[int, Optional[TimesheetReviewResponse], Optional[TaskReviewFailure]]:
    """Run the review pipeline for one task within a bulk request.

    Never raises - any failure (expected or unexpected) is converted into
    a safe TaskReviewFailure so that one bad task cannot stop the rest of
    the batch from being processed.
    """
    task_id = task_group.backlog_task.task_id
    logger.info("[%s] Task start | task_id=%s", bulk_id, task_id)

    try:
        single_request = build_task_review_request(bulk_request, task_group)
        response = process_single_task_review(single_request)
        logger.info(
            "[%s] Task completed | task_id=%s decision=%s",
            bulk_id,
            task_id,
            response.decision.value,
        )
        return index, response, None

    except TaskReviewError as exc:
        logger.error(
            "[%s] Task failed | task_id=%s error_type=%s",
            bulk_id,
            task_id,
            exc.error_type,
        )
        return index, None, TaskReviewFailure(
            task_id=task_id, error_type=exc.error_type, error_message=exc.message
        )

    except Exception:  # noqa: BLE001 - deliberately broad: isolate this task's failure
        logger.exception(
            "[%s] Task failed with an unexpected error | task_id=%s", bulk_id, task_id
        )
        return index, None, TaskReviewFailure(
            task_id=task_id,
            error_type="UnexpectedFailure",
            error_message=(
                "The task could not be reviewed due to an unexpected error."
            ),
        )


@app.post(
    "/api/v1/manager/timesheet-review/bulk",
    response_model=BulkTimesheetReviewResponse,
    tags=["Manager Review"],
    summary="Review multiple backlog tasks for one employee in a single request",
)
def review_timesheet_bulk(
    request: BulkTimesheetReviewRequest,
) -> BulkTimesheetReviewResponse:
    """Review multiple backlog tasks for one employee, each independently.

    Every task is sent to the LLM as its own isolated review (never
    combined with other tasks in one prompt), using the exact same
    pipeline as the single-task endpoint via ``process_single_task_review``.
    A failure on one task never stops the others from being processed.
    All aggregate totals are computed locally in Python, never by the LLM.
    """
    bulk_id = uuid4().hex[:8]
    started_at = time.perf_counter()
    logger.info(
        "[%s] Starting bulk review | employee=%s tasks=%d",
        bulk_id,
        request.employee.employee_id,
        len(request.tasks),
    )

    results_by_index: Dict[int, TimesheetReviewResponse] = {}
    failures_by_index: Dict[int, TaskReviewFailure] = {}

    def _run_sequentially() -> None:
        results_by_index.clear()
        failures_by_index.clear()
        for i, task_group in enumerate(request.tasks):
            idx, response, failure = _process_one_bulk_task(
                bulk_id, i, request, task_group
            )
            if response is not None:
                results_by_index[idx] = response
            else:
                failures_by_index[idx] = failure  # type: ignore[assignment]

    max_workers = min(BULK_MAX_WORKERS, len(request.tasks))
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_one_bulk_task, bulk_id, i, request, task_group)
                for i, task_group in enumerate(request.tasks)
            ]
            for future in as_completed(futures):
                idx, response, failure = future.result()
                if response is not None:
                    results_by_index[idx] = response
                else:
                    failures_by_index[idx] = failure  # type: ignore[assignment]

    except Exception:  # noqa: BLE001 - parallel execution failed; fall back safely
        logger.exception(
            "[%s] Parallel bulk processing failed; falling back to sequential.",
            bulk_id,
        )
        _run_sequentially()

    # Preserve input task order within each of the two result lists.
    ordered_results = [results_by_index[i] for i in sorted(results_by_index)]
    ordered_failures = [failures_by_index[i] for i in sorted(failures_by_index)]

    total_tasks = len(request.tasks)
    completed_tasks = len(ordered_results)
    failed_tasks = len(ordered_failures)
    accepted_tasks = sum(1 for r in ordered_results if r.decision == Decision.ACCEPT)
    review_tasks = sum(1 for r in ordered_results if r.decision == Decision.REVIEW)
    correction_required_tasks = sum(
        1 for r in ordered_results if r.decision == Decision.CORRECTION_REQUIRED
    )
    total_planned_hours = round(
        sum(r.effort_summary.total_planned_hours for r in ordered_results), 2
    )
    total_actual_hours = round(
        sum(r.effort_summary.total_actual_hours for r in ordered_results), 2
    )

    if failed_tasks == 0:
        bulk_status = "COMPLETED"
    elif completed_tasks == 0:
        bulk_status = "FAILED"
    else:
        bulk_status = "PARTIALLY_COMPLETED"

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    logger.info(
        "[%s] Bulk review completed | status=%s completed=%d failed=%d elapsed_ms=%s",
        bulk_id,
        bulk_status,
        completed_tasks,
        failed_tasks,
        elapsed_ms,
    )

    return BulkTimesheetReviewResponse(
        status=bulk_status,
        employee_id=request.employee.employee_id,
        employee_name=request.employee.employee_name,
        report_type=request.report_type,
        review_period=ReviewPeriod(
            start_date=request.start_date, end_date=request.end_date
        ),
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
        accepted_tasks=accepted_tasks,
        review_tasks=review_tasks,
        correction_required_tasks=correction_required_tasks,
        total_planned_hours=total_planned_hours,
        total_actual_hours=total_actual_hours,
        results=ordered_results,
        failures=ordered_failures,
    )


# ==============================================================
# 17. SWAGGER DOCUMENTATION
# ==============================================================
# Swagger UI is auto-generated by FastAPI and available at /docs
# ReDoc is auto-generated by FastAPI and available at /redoc


# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)