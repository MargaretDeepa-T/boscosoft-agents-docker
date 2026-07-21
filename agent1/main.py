from __future__ import annotations

import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from groq import Groq
from pydantic import BaseModel, Field


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()
BULK_MAX_WORKERS = max(
    1,
    int(os.getenv("BULK_MAX_WORKERS", "5")),
)

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(
    title="Boscosoft Task Validation API",
    version="1.2.0",
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

    return f"""
Evaluate this backlog task:

{task.model_dump_json(indent=2)}

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
- If the original description is correct, return exactly: {json.dumps(task.task_description)}
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


def request_groq_validation(
    task: TaskInput,
    user_prompt: str,
) -> dict[str, Any]:
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

    raw = response.choices[0].message.content or ""

    if not raw:
        raise RuntimeError("Groq returned an empty response")

    parsed = json.loads(raw)

    if not isinstance(parsed, dict):
        raise RuntimeError("Groq response must be a JSON object")

    return parsed


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

    except Exception as error:
        return validation_error_result(task, error)



def validate_tasks_concurrently(
    tasks: list[TaskInput],
    project: str | None = None,
    sprint: str | None = None,
) -> list[ValidationResult]:
    """Validate tasks concurrently while preserving the input order."""
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