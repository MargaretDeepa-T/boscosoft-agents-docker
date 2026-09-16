"""Tests for CANNOT_VALIDATE_ESTIMATE across single, bulk JSON and bulk Excel.

Groq is mocked, so these run offline and deterministically.
Run:  GROQ_API_KEY=dummy pytest -q test_cannot_validate.py
"""
import io
import os

os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")
os.environ.pop("REDIS_URL", None)

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

GOOD_DESC_1 = (
    "Build a REST endpoint that returns paginated parish member records "
    "filtered by diocese, with sorting by name and join date, plus unit tests."
)
GOOD_DESC_2 = (
    "Implement JWT login with refresh token rotation, lockout after five "
    "failed attempts, and audit logging of every authentication event."
)
BAD_DESC_EXAMPLE = "Front end design for dashboard in CHMS parish portal.etc"


def fake_llm(task, prompt):
    """Mimics Groq. Reasonable estimates -> PROCEED, 1h -> REVIEW_ESTIMATE.
    'LLM-VAGUE' tasks -> CANNOT_VALIDATE_ESTIMATE *with a number* to prove
    the server nulls it regardless of what the model returns."""
    base = {
        "task_id": task.task_id,
        "task_title_assessment": "Clear.",
        "task_description_assessment": "Clear.",
        "scope_assessment": "Appropriate.",
        "effort_assessment": "Reasonable.",
        "suggested_task_title": task.task_title,
        "suggested_task_description": task.task_description,
        "confidence_score": 0.85,
        "recommendation": "Looks fine.",
    }
    if task.task_id.startswith("LLM-VAGUE"):
        return {**base, "decision": "CANNOT_VALIDATE_ESTIMATE",
                "task_description_assessment": "Too vague.",
                "suggested_estimated_hours": task.estimated_hours}
    if task.estimated_hours <= 1:
        return {**base, "decision": "REVIEW_ESTIMATE",
                "suggested_estimated_hours": 16}
    return {**base, "decision": "PROCEED",
            "suggested_estimated_hours": task.estimated_hours}


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    main.clear_validation_cache()
    calls = []

    def _fake(task, prompt):
        calls.append(task.task_id)
        return fake_llm(task, prompt)

    monkeypatch.setattr(main, "request_groq_validation", _fake)
    yield calls
    main.clear_validation_cache()


def assert_cannot(result):
    assert result["decision"] == "CANNOT_VALIDATE_ESTIMATE"
    assert result["suggested_estimated_hours"] is None
    assert result["recommendation"] == main.INSUFFICIENT_DESCRIPTION_RECOMMENDATION


def excel_bytes(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def upload(rows):
    return client.post(
        "/api/v1/backlog/upload-and-validate",
        files={"file": ("backlog.xlsx", excel_bytes(rows),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )


# ---------------------------------------------------------------- TEST 1
def test_1_single_insufficient_returns_null(setup):
    r = client.post("/api/v1/task/validate", json={
        "task_id": "TASK-001",
        "task_title": "Front end design for dashboard",
        "task_description": BAD_DESC_EXAMPLE,
        "estimated_hours": 50,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["result"]["task_id"] == "TASK-001"
    assert_cannot(body["result"])
    assert setup == []  # no LLM call spent


# ---------------------------------------------------------------- TEST 2
def test_2_single_sufficient_returns_numeric():
    r = client.post("/api/v1/task/validate", json={
        "task_id": "TASK-002",
        "task_title": "Member listing API",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 12,
    })
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["result"]["decision"] == "PROCEED"
    assert isinstance(body["result"]["suggested_estimated_hours"], (int, float))
    assert body["result"]["suggested_estimated_hours"] == 12


# ---------------------------------------------------------------- TEST 3
def test_3_bulk_json_mixed():
    r = client.post("/api/v1/backlog/validate", json={
        "project": "CHMS", "sprint": "S1",
        "tasks": [
            {"task_id": "T1", "task_title": "Front end design for dashboard",
             "task_description": BAD_DESC_EXAMPLE, "estimated_hours": 50},
            {"task_id": "T2", "task_title": "Member listing API",
             "task_description": GOOD_DESC_1, "estimated_hours": 12},
            {"task_id": "T3", "task_title": "Fix bug",
             "task_description": "", "estimated_hours": 4},
            {"task_id": "T4", "task_title": "Login",
             "task_description": GOOD_DESC_2, "estimated_hours": 1},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    res = {x["task_id"]: x for x in body["results"]}
    assert [x["task_id"] for x in body["results"]] == ["T1", "T2", "T3", "T4"]
    assert_cannot(res["T1"])
    assert res["T2"]["decision"] == "PROCEED"
    assert res["T2"]["suggested_estimated_hours"] == 12
    assert_cannot(res["T3"])
    assert res["T4"]["decision"] == "REVIEW_ESTIMATE"
    assert res["T4"]["suggested_estimated_hours"] == 16
    assert body["status"] == "COMPLETED"
    assert body["total_tasks"] == 4
    assert body["validated_tasks"] == 4
    assert body["failed_tasks"] == 0
    assert body["insufficient_description_tasks"] == 2


# ---------------------------------------------------------------- TEST 4
def test_4_excel_mixed():
    r = upload([
        {"Task ID": "X1", "Task": "Front end design for dashboard",
         "Description": BAD_DESC_EXAMPLE, "Estimated Hours": 50},
        {"Task ID": "X2", "Task": "Member listing API",
         "Description": GOOD_DESC_1, "Estimated Hours": 12},
        {"Task ID": "X3", "Task": "UI", "Description": "UI", "Estimated Hours": 8},
        {"Task ID": "X4", "Task": "Login",
         "Description": GOOD_DESC_2, "Estimated Hours": 1},
    ])
    assert r.status_code == 200, r.text
    body = r.json()
    res = {x["task_id"]: x for x in body["results"]}
    assert_cannot(res["X1"])          # manual 50 must NOT leak
    assert res["X2"]["suggested_estimated_hours"] == 12
    assert_cannot(res["X3"])
    assert res["X4"]["decision"] == "REVIEW_ESTIMATE"
    assert res["X4"]["suggested_estimated_hours"] == 16
    assert body["status"] == "COMPLETED"
    assert body["validated_tasks"] == 4 and body["failed_tasks"] == 0


# ---------------------------------------------------------------- TEST 5
def test_5_excel_all_insufficient(setup):
    r = upload([
        {"Task": "Front end design for dashboard",
         "Description": BAD_DESC_EXAMPLE, "Estimated Hours": 50},
        {"Task": "API", "Description": "", "Estimated Hours": 10},
        {"Task": "Testing", "Description": "Testing", "Estimated Hours": 5},
        {"Task": "Update module", "Description": "N/A", "Estimated Hours": 3},
    ])
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 4
    for result in body["results"]:
        assert_cannot(result)
    # A valid business decision - not a failed bulk request.
    assert body["status"] == "COMPLETED"
    assert body["validated_tasks"] == 4
    assert body["failed_tasks"] == 0
    assert body["insufficient_description_tasks"] == 4
    assert setup == []


# ---------------------------------------------------- extra guard rails
def test_llm_returned_cannot_with_number_is_nulled():
    """Description passes the pre-check, but the LLM says CANNOT and
    (wrongly) returns the manual hours. Server must null it, and must
    not pin a canonical estimate."""
    r = client.post("/api/v1/task/validate", json={
        "task_id": "LLM-VAGUE-1", "task_title": "Portal improvements",
        "task_description": "Various improvements across several different portal areas as discussed earlier",
        "estimated_hours": 40,
    })
    assert_cannot(r.json()["result"])
    assert main._canonical_cache == {}


def test_stale_cache_entry_with_number_is_nulled():
    """Results cached by the old version carried the manual hours
    forward. Loading them must still yield null."""
    task = main.TaskInput(task_id="OLD-1", task_title="Member listing API",
                          task_description=GOOD_DESC_1, estimated_hours=30)
    fp = main.compute_task_fingerprint(task)
    legacy_json = (
        '{"task_id":"OLD-1","decision":"CANNOT_VALIDATE_ESTIMATE",'
        '"task_title_assessment":"x","task_description_assessment":"x",'
        '"scope_assessment":"x","effort_assessment":"x",'
        '"suggested_task_title":"t","suggested_task_description":"d",'
        '"suggested_estimated_hours":30,"confidence_score":0.5,'
        '"recommendation":"old"}'
    )
    main._validation_cache[fp] = main.ValidationResult.model_validate_json(legacy_json)
    result = main.validate_with_groq(task)
    assert result.decision == "CANNOT_VALIDATE_ESTIMATE"
    assert result.suggested_estimated_hours is None


def test_non_cannot_decision_still_requires_number():
    with pytest.raises(Exception):
        main.ValidationResult(
            task_id="Z", decision="PROCEED", task_title_assessment="a",
            task_description_assessment="a", scope_assessment="a",
            effort_assessment="a", suggested_task_title="a",
            suggested_task_description="a", suggested_estimated_hours=None,
            confidence_score=0.5, recommendation="a",
        )


def test_error_results_still_count_as_failed(monkeypatch):
    def boom(task, prompt):
        raise RuntimeError("groq down")
    monkeypatch.setattr(main, "request_groq_validation", boom)
    r = client.post("/api/v1/backlog/validate", json={"tasks": [
        {"task_id": "E1", "task_title": "Member listing API",
         "task_description": GOOD_DESC_1, "estimated_hours": 12},
        {"task_id": "E2", "task_title": "API",
         "task_description": "", "estimated_hours": 3},
    ]})
    body = r.json()
    assert body["status"] == "PARTIALLY_COMPLETED"
    assert body["failed_tasks"] == 1 and body["validated_tasks"] == 1
    res = {x["task_id"]: x for x in body["results"]}
    assert res["E1"]["decision"] == "ERROR"
    assert_cannot(res["E2"])


def test_canonical_path_unchanged():
    """Second validation of same content with different hours uses the
    pinned canonical estimate (no second LLM call) - existing behaviour."""
    payload = {"task_id": "C1", "task_title": "Member listing API",
               "task_description": GOOD_DESC_1, "estimated_hours": 12}
    client.post("/api/v1/task/validate", json=payload)
    r = client.post("/api/v1/task/validate", json={**payload, "estimated_hours": 60})
    result = r.json()["result"]
    assert result["decision"] == "REVIEW_ESTIMATE"
    assert result["suggested_estimated_hours"] == 12


# ------------------------------------------------ Excel column aliases
@pytest.mark.parametrize("header", [
    "Manual Estimate", "manual_estimate", "Manual Hours",
    "Estimate", "Estimated Hours", "Est Hrs",
])
def test_estimate_column_aliases(header):
    df = pd.DataFrame([{"Task": "Member listing API",
                        "Description": GOOD_DESC_1, header: 12}])
    tasks = main.dataframe_to_tasks(df)
    assert tasks[0].estimated_hours == 12


def test_excel_manual_estimate_column_end_to_end():
    r = upload([
        {"Task": "Front end design for dashboard",
         "Description": BAD_DESC_EXAMPLE, "Manual Estimate": 50},
        {"Task": "Member listing API",
         "Description": GOOD_DESC_1, "Manual Estimate": 12},
    ])
    res = r.json()["results"]
    assert_cannot(res[0])
    assert res[1]["decision"] == "PROCEED"
    assert res[1]["suggested_estimated_hours"] == 12  # read from Manual Estimate, not 0


def test_prompt_contains_title_consistency_rule():
    assert "TITLE CONSISTENCY (MANDATORY)" in main.SYSTEM_PROMPT
    task = main.TaskInput(task_id="P", task_title="t",
                          task_description=GOOD_DESC_1, estimated_hours=5)
    assert "acceptable as-is" in main.build_prompt(task)
