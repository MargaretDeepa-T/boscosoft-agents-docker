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

import json
import logging
import os
import time
from datetime import date
from enum import Enum
from typing import List, Optional
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

MAX_LLM_RETRIES = 3
GROQ_REQUEST_TIMEOUT_SECONDS = 60
LLM_TEMPERATURE = 0.1
RETRY_BACKOFF_SECONDS = 1.5

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
                "activity_description": entry.activity_description,
                "planned_hours": entry.planned_hours,
                "actual_hours": entry.actual_hours,
            }
            for entry in request.timesheet_entries
        ],
        indent=2,
    )

    backlog_task = request.backlog_task

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
Task Description: {backlog_task.task_description}
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


class GroqTransientFailure(Exception):
    """Timeout, connection, or 5xx error from Groq. Safe to retry."""


class GroqPermanentFailure(Exception):
    """Any other non-retryable Groq API error (e.g. a 4xx bad request)."""


groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_REQUEST_TIMEOUT_SECONDS)


def call_groq_llm(prompt: str) -> str:
    """Call the Groq chat completion API and return the raw text response.

    Raises:
        GroqAuthenticationFailure: invalid/missing credentials.
        GroqTransientFailure: timeout, connection, or 5xx error - the
            caller may safely retry.
        GroqPermanentFailure: any other non-retryable API error.
    """
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
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content or ""

    except AuthenticationError as exc:
        logger.error("Groq authentication error: %s", exc)
        raise GroqAuthenticationFailure(str(exc)) from exc

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
        HTTPException: 401 on authentication failure, 502 on a permanent
        API error or once retries are exhausted.
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
            raise HTTPException(
                status_code=401,
                detail="Authentication with the Groq API failed. Check GROQ_API_KEY.",
            ) from exc

        except GroqPermanentFailure as exc:
            raise HTTPException(
                status_code=502,
                detail=f"The Groq API returned a non-retryable error: {exc}",
            ) from exc

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
        "LLM failed to produce a valid review after %d attempts.", MAX_LLM_RETRIES
    )
    raise HTTPException(
        status_code=502,
        detail=(
            f"The AI model failed to return a valid review after "
            f"{MAX_LLM_RETRIES} attempts: {last_error}"
        ),
    )


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
        HTTPException: 502 if the merged payload fails schema validation.
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
        raise HTTPException(
            status_code=502,
            detail=f"The AI model response failed schema validation: {exc}",
        ) from exc


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

    effort_summary = calculate_effort_summary(
        request.backlog_task, request.timesheet_entries
    )
    prompt = build_prompt(request, effort_summary)
    llm_result = get_llm_review(prompt)
    response = assemble_response(request, llm_result, effort_summary)

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)
    logger.info(
        "[%s] Completed review | decision=%s elapsed_ms=%s",
        review_id,
        response.decision.value,
        elapsed_ms,
    )
    return response


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
