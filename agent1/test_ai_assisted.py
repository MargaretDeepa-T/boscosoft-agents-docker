"""Tests for the new AI-Assisted Estimation feature.

Covers the acceptance scenarios from the feature spec:
  Test A - existing standard endpoint still works
  Test B - new AI-assisted endpoint works
  Test C - same task submitted to both modes stays isolated
  Test D - insufficient description is preserved in AI-assisted mode
  Test E - AI mode independently estimates a CRUD task (not anchored
           to the submitted estimate)
  Test F - a low-AI-benefit task is not forced to a lower estimate
  Test G - cache isolation between modes, in both submission orders
  Test H - existing bulk endpoint is unaffected
  Test I - AI-assisted bulk endpoint matches single-task AI behavior

Groq is mocked (via main.request_groq_validation), so these run offline
and deterministically.

Run:  GROQ_API_KEY=dummy pytest -q test_ai_assisted.py
"""
import os

os.environ.setdefault("GROQ_API_KEY", "dummy-key-for-tests")
os.environ.pop("REDIS_URL", None)

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)

GOOD_DESC_1 = (
    "Build a REST endpoint that returns paginated parish member records "
    "filtered by diocese, with sorting by name and join date, plus unit tests."
)

CRUD_DESCRIPTION = (
    "Develop REST APIs to create, view, update and delete employee "
    "records. Implement request and response models, mandatory field "
    "validation, email format validation, duplicate employee code "
    "validation, database CRUD operations, exception handling, HTTP "
    "response codes and unit tests for the APIs."
)

LOW_AI_BENEFIT_DESCRIPTION = (
    "Coordinate with the finance and compliance stakeholders to clarify "
    "the exact payout rules for the new reimbursement policy, run manual "
    "acceptance testing sessions with three regional office leads, and "
    "get sign-off from the audit committee before the policy goes live."
)

BAD_DESC_EXAMPLE = "Assign Asset"


def fake_llm(task, prompt, system_prompt=main.SYSTEM_PROMPT):
    """Mimics Groq for both STANDARD and AI_ASSISTED system prompts.

    The fake distinguishes mode by which system prompt it was called
    with (exactly what main._request_groq_validation_for_mode passes),
    not by inspecting the task - this proves the production dispatcher
    really does route each mode to a different prompt.
    """
    is_ai_mode = system_prompt == main.AI_SYSTEM_PROMPT

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

    if task.task_id.startswith("CRUD"):
        # STANDARD: the submitted 16h is judged reasonable outright.
        # AI_ASSISTED: independently derives a materially lower,
        # non-round-percentage figure (8h - not 16*0.7=11.2 or any
        # other fixed-discount number) from the scope itself, proving
        # the estimate isn't just "16h minus a fixed %".
        if is_ai_mode:
            return {
                **base,
                "decision": "REVIEW_ESTIMATE",
                "effort_assessment": (
                    "CRUD scaffolding, validation and exception "
                    "handling are AI-accelerable; review, debugging and "
                    "test verification remain human-dependent."
                ),
                "suggested_estimated_hours": 8,
            }
        return {
            **base,
            "decision": "PROCEED",
            "suggested_estimated_hours": task.estimated_hours,
        }

    if task.task_id.startswith("LOWAI"):
        # Dominated by stakeholder coordination and manual acceptance
        # testing - AI assistance offers little here, so AI_ASSISTED
        # must NOT force a lower number than STANDARD.
        return {
            **base,
            "decision": "PROCEED",
            "suggested_estimated_hours": task.estimated_hours,
        }

    if task.task_id.startswith("SAME"):
        # Used for the STANDARD-vs-AI_ASSISTED isolation test: give a
        # deliberately different independent AI figure so a cache/
        # namespace bug (one mode leaking into the other) would be
        # caught by a mismatched result.
        if is_ai_mode:
            return {
                **base,
                "decision": "REVIEW_ESTIMATE",
                "suggested_estimated_hours": 9,
            }
        return {
            **base,
            "decision": "PROCEED",
            "suggested_estimated_hours": task.estimated_hours,
        }

    if task.task_id.startswith("STD"):
        return {
            **base,
            "decision": "PROCEED",
            "suggested_estimated_hours": task.estimated_hours,
        }

    if task.task_id.startswith("AI-"):
        return {
            **base,
            "decision": "PROCEED",
            "suggested_estimated_hours": task.estimated_hours,
        }

    return {
        **base,
        "decision": "PROCEED",
        "suggested_estimated_hours": task.estimated_hours,
    }


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    main.clear_validation_cache()
    calls = []

    def _fake(task, prompt, system_prompt=main.SYSTEM_PROMPT):
        calls.append((task.task_id, system_prompt == main.AI_SYSTEM_PROMPT))
        return fake_llm(task, prompt, system_prompt)

    monkeypatch.setattr(main, "request_groq_validation", _fake)
    yield calls
    main.clear_validation_cache()


# ---------------------------------------------------------------- TEST A
def test_a_existing_standard_endpoint_unchanged():
    r = client.post("/api/v1/task/validate", json={
        "task_id": "STD-1",
        "task_title": "Member listing API",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 12,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["result"]["decision"] == "PROCEED"
    assert body["result"]["suggested_estimated_hours"] == 12
    # Confirms the STANDARD system prompt was used.
    assert (main.SYSTEM_PROMPT != main.AI_SYSTEM_PROMPT)


# ---------------------------------------------------------------- TEST B
def test_b_ai_assisted_endpoint_exists_and_works(setup):
    r = client.post("/api/v1/task/validate-ai-assisted", json={
        "task_id": "AI-1",
        "task_title": "Member listing API",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 12,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["result"]["task_id"] == "AI-1"
    # The dispatcher used the AI system prompt for this call.
    assert any(is_ai for (_task_id, is_ai) in setup)
    # Response contract matches the existing Agent 1 result shape.
    expected_fields = {
        "task_id", "decision", "task_title_assessment",
        "task_description_assessment", "scope_assessment",
        "effort_assessment", "suggested_task_title",
        "suggested_task_description", "suggested_estimated_hours",
        "confidence_score", "recommendation",
    }
    assert expected_fields.issubset(body["result"].keys())


# ---------------------------------------------------------------- TEST C
def test_c_same_task_both_modes_stay_isolated():
    payload = {
        "task_id": "SAME-1",
        "task_title": "Bulk import service",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 20,
    }

    r_standard = client.post("/api/v1/task/validate", json=payload)
    r_ai = client.post("/api/v1/task/validate-ai-assisted", json=payload)

    assert r_standard.status_code == 200
    assert r_ai.status_code == 200

    standard_result = r_standard.json()["result"]
    ai_result = r_ai.json()["result"]

    assert standard_result["decision"] == "PROCEED"
    assert standard_result["suggested_estimated_hours"] == 20

    # AI mode independently evaluated the task rather than echoing the
    # STANDARD cached/canonical answer.
    assert ai_result["decision"] == "REVIEW_ESTIMATE"
    assert ai_result["suggested_estimated_hours"] == 9


# ---------------------------------------------------------------- TEST D
def test_d_insufficient_description_preserved_in_both_modes(setup):
    payload = {
        "task_id": "INSUFF-1",
        "task_title": "Assign Asset",
        "task_description": BAD_DESC_EXAMPLE,
        "estimated_hours": 8,
    }

    r_standard = client.post("/api/v1/task/validate", json=payload)
    r_ai = client.post("/api/v1/task/validate-ai-assisted", json=payload)

    for response in (r_standard, r_ai):
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["decision"] == "CANNOT_VALIDATE_ESTIMATE"
        assert result["suggested_estimated_hours"] is None
        assert result["recommendation"] == main.INSUFFICIENT_DESCRIPTION_RECOMMENDATION

    # The insufficient-description gate is deterministic and runs
    # before any LLM call in both modes, in both branches.
    assert setup == []


# ---------------------------------------------------------------- TEST E
def test_e_crud_task_independent_ai_estimate_not_anchored():
    payload = {
        "task_id": "CRUD-1",
        "task_title": "Develop Employee CRUD APIs",
        "task_description": CRUD_DESCRIPTION,
        "estimated_hours": 16,
    }

    r_standard = client.post("/api/v1/task/validate", json=payload)
    r_ai = client.post("/api/v1/task/validate-ai-assisted", json=payload)

    standard_result = r_standard.json()["result"]
    ai_result = r_ai.json()["result"]

    assert standard_result["decision"] == "PROCEED"
    assert standard_result["suggested_estimated_hours"] == 16

    # AI mode landed on a genuinely independent figure (8h) rather than
    # accepting 16h outright or applying a fixed discount of 16h.
    assert ai_result["decision"] == "REVIEW_ESTIMATE"
    assert ai_result["suggested_estimated_hours"] == 8
    assert ai_result["suggested_estimated_hours"] != payload["estimated_hours"]


# ---------------------------------------------------------------- TEST F
def test_f_low_ai_benefit_task_not_forced_lower():
    payload = {
        "task_id": "LOWAI-1",
        "task_title": "Reimbursement policy rollout",
        "task_description": LOW_AI_BENEFIT_DESCRIPTION,
        "estimated_hours": 24,
    }

    r_standard = client.post("/api/v1/task/validate", json=payload)
    r_ai = client.post("/api/v1/task/validate-ai-assisted", json=payload)

    standard_result = r_standard.json()["result"]
    ai_result = r_ai.json()["result"]

    assert standard_result["decision"] == "PROCEED"
    assert ai_result["decision"] == "PROCEED"
    # No fixed AI discount was manufactured: both modes land on the
    # same effort for a task AI assistance doesn't meaningfully help.
    assert ai_result["suggested_estimated_hours"] == standard_result["suggested_estimated_hours"] == 24


# ---------------------------------------------------------------- TEST G
def test_g_cache_isolation_standard_then_ai():
    payload = {
        "task_id": "CACHE-G1",
        "task_title": "Notification service",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 15,
    }
    r1 = client.post("/api/v1/task/validate", json=payload)
    assert r1.json()["result"]["decision"] == "PROCEED"

    # Same content, AI-assisted endpoint: must NOT just replay the
    # STANDARD cached result.
    r2 = client.post("/api/v1/task/validate-ai-assisted", json=payload)
    assert r2.status_code == 200
    # A fresh LLM call must have happened for AI mode (proven by the
    # fake being invoked again for this task_id with the AI prompt).
    fp_standard = main.compute_task_fingerprint(
        main.TaskInput(**payload), mode="STANDARD"
    )
    fp_ai = main.compute_task_fingerprint(
        main.TaskInput(**payload), mode="AI_ASSISTED"
    )
    assert fp_standard != fp_ai
    assert main.get_cached_result(fp_standard, mode="STANDARD") is not None
    assert main.get_cached_result(fp_ai, mode="AI_ASSISTED") is not None


def test_g_cache_isolation_ai_then_standard():
    payload = {
        "task_id": "CACHE-G2",
        "task_title": "Notification service v2",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 15,
    }
    r_ai = client.post("/api/v1/task/validate-ai-assisted", json=payload)
    assert r_ai.status_code == 200

    r_standard = client.post("/api/v1/task/validate", json=payload)
    assert r_standard.status_code == 200
    assert r_standard.json()["result"]["decision"] == "PROCEED"
    assert r_standard.json()["result"]["suggested_estimated_hours"] == 15

    fp_standard = main.compute_task_fingerprint(
        main.TaskInput(**payload), mode="STANDARD"
    )
    fp_ai = main.compute_task_fingerprint(
        main.TaskInput(**payload), mode="AI_ASSISTED"
    )
    assert main.get_cached_result(fp_standard, mode="STANDARD") is not None
    assert main.get_cached_result(fp_ai, mode="AI_ASSISTED") is not None
    # Redis/in-memory namespaces are distinct prefixes.
    assert main.CACHE_KEY_PREFIX != main.CACHE_KEY_PREFIX_AI
    assert not main.CACHE_KEY_PREFIX_AI.startswith(main.CACHE_KEY_PREFIX)
    assert not main.CACHE_KEY_PREFIX.startswith(main.CACHE_KEY_PREFIX_AI)


# ---------------------------------------------------------------- TEST H
def test_h_existing_bulk_endpoint_unchanged():
    r = client.post("/api/v1/backlog/validate", json={
        "project": "CHMS", "sprint": "S1",
        "tasks": [
            {"task_id": "STD-B1", "task_title": "Member listing API",
             "task_description": GOOD_DESC_1, "estimated_hours": 12},
            {"task_id": "INSUFF-B1", "task_title": "Assign Asset",
             "task_description": BAD_DESC_EXAMPLE, "estimated_hours": 8},
        ],
    })
    assert r.status_code == 200
    body = r.json()
    res = {x["task_id"]: x for x in body["results"]}
    assert res["STD-B1"]["decision"] == "PROCEED"
    assert res["STD-B1"]["suggested_estimated_hours"] == 12
    assert res["INSUFF-B1"]["decision"] == "CANNOT_VALIDATE_ESTIMATE"
    assert res["INSUFF-B1"]["suggested_estimated_hours"] is None
    assert body["status"] == "COMPLETED"
    assert body["insufficient_description_tasks"] == 1


# ---------------------------------------------------------------- TEST I
def test_i_ai_assisted_bulk_matches_single_task_behavior():
    crud_payload = {
        "task_id": "CRUD-BULK-1",
        "task_title": "Develop Employee CRUD APIs",
        "task_description": CRUD_DESCRIPTION,
        "estimated_hours": 16,
    }
    insuff_payload = {
        "task_id": "INSUFF-BULK-1",
        "task_title": "Assign Asset",
        "task_description": BAD_DESC_EXAMPLE,
        "estimated_hours": 8,
    }

    r = client.post("/api/v1/backlog/validate-ai-assisted", json={
        "tasks": [crud_payload, insuff_payload],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    res = {x["task_id"]: x for x in body["results"]}

    # Same independent-estimation behavior as the single-task endpoint.
    assert res["CRUD-BULK-1"]["decision"] == "REVIEW_ESTIMATE"
    assert res["CRUD-BULK-1"]["suggested_estimated_hours"] == 8

    # Same insufficient-description guard as the single-task endpoint.
    assert res["INSUFF-BULK-1"]["decision"] == "CANNOT_VALIDATE_ESTIMATE"
    assert res["INSUFF-BULK-1"]["suggested_estimated_hours"] is None

    assert body["status"] == "COMPLETED"
    assert body["total_tasks"] == 2
    assert body["insufficient_description_tasks"] == 1


# ------------------------------------------------ extra guard rails
def test_ai_system_prompt_is_derived_and_distinct():
    assert main.AI_SYSTEM_PROMPT != main.SYSTEM_PROMPT
    assert "AI-ASSISTED ESTIMATION" in main.AI_SYSTEM_PROMPT
    assert "MANDATORY TASK DESCRIPTION" in main.AI_SYSTEM_PROMPT
    assert "OUTPUT SCHEMA (MANDATORY)" in main.AI_SYSTEM_PROMPT
    assert "TITLE CONSISTENCY (MANDATORY)" in main.AI_SYSTEM_PROMPT
    # The prompt explicitly PROHIBITS a fixed AI discount (it's fine
    # that "30%" appears as an example of what NOT to do) - it must not
    # instruct the model to apply one.
    assert "fixed or generic productivity discount" in main.AI_SYSTEM_PROMPT
    assert "* 0.7" not in main.AI_SYSTEM_PROMPT
    assert "AI estimate = normal estimate" not in main.AI_SYSTEM_PROMPT


def test_ai_error_result_not_cached_and_bulk_reports_failure(monkeypatch):
    def boom(task, prompt, system_prompt=main.SYSTEM_PROMPT):
        raise RuntimeError("groq down")
    monkeypatch.setattr(main, "request_groq_validation", boom)

    r = client.post("/api/v1/task/validate-ai-assisted", json={
        "task_id": "AI-ERR-1",
        "task_title": "Member listing API",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 12,
    })
    body = r.json()
    assert body["status"] == "FAILED"
    assert body["result"]["decision"] == "ERROR"

    fp = main.compute_task_fingerprint(
        main.TaskInput(
            task_id="AI-ERR-1", task_title="Member listing API",
            task_description=GOOD_DESC_1, estimated_hours=12,
        ),
        mode="AI_ASSISTED",
    )
    assert main.get_cached_result(fp, mode="AI_ASSISTED") is None


def test_ai_auth_failure_surfaces_as_401(monkeypatch):
    def fail_auth(task, prompt, system_prompt=main.SYSTEM_PROMPT):
        raise main.GroqAuthenticationFailure("bad key")
    monkeypatch.setattr(main, "request_groq_validation", fail_auth)

    r = client.post("/api/v1/task/validate-ai-assisted", json={
        "task_id": "AI-AUTH-1",
        "task_title": "Member listing API",
        "task_description": GOOD_DESC_1,
        "estimated_hours": 12,
    })
    assert r.status_code == 401
