"""
tests/test_rewards.py
=====================
Unit tests for RewardEngine.
Verifies reward values for correct, incorrect, and invalid actions.

Run with: pytest tests/test_rewards.py -v
"""

import pytest
from app.rewards import RewardEngine
from app.models import (
    Action, ActionType, BillingState, BloodBankState,
    Difficulty, InternalState, ReportState, TaskType,
)

engine = RewardEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_state(correct: str, submitted: str = None, classified: bool = False):
    return InternalState(
        episode_id="x", scenario_id="report_easy",
        task_type=TaskType.REPORT_CLASSIFICATION,
        difficulty=Difficulty.EASY, step_number=1, max_steps=3,
        report_state=ReportState(
            report_id="R1", correct_label=correct,
            submitted_label=submitted, classified=classified,
        ),
    )


def _billing_state(**kw):
    defaults = dict(
        claim_id="C1", correct_code="85025", code_is_valid=True,
        insurance_covers=True, has_discrepancy=False,
        correct_final_decision="approve_claim",
        code_validated=False, code_correct_result=None,
        insurance_verified=False, insurance_correct_result=None,
        discrepancy_flagged=False, final_decision=None,
    )
    defaults.update(kw)
    return InternalState(
        episode_id="x", scenario_id="billing_easy",
        task_type=TaskType.BILLING_VERIFICATION,
        difficulty=Difficulty.EASY, step_number=1, max_steps=6,
        billing_state=BillingState(**defaults),
    )


def _blood_state(**kw):
    defaults = dict(
        request_id="B1", requested_type="A+", requested_units=2,
        urgency="routine",
        inventory={"A+": 10, "O-": 5},
        initial_inventory={"A+": 12, "O-": 5},
        expiry_data={},
        should_discard_types=[], should_request_restock=False,
        correct_allocation_type="A+",
        restock_requested=False, allocation_fulfilled=False,
        allocated_type=None, allocated_units=0, units_discarded=0,
    )
    defaults.update(kw)
    return InternalState(
        episode_id="x", scenario_id="bloodbank_easy",
        task_type=TaskType.BLOOD_BANK,
        difficulty=Difficulty.EASY, step_number=1, max_steps=5,
        blood_bank_state=BloodBankState(**defaults),
    )


def _action(atype, **payload):
    return Action(action_type=atype, payload=payload, episode_id="x")


# ---------------------------------------------------------------------------
# Report rewards
# ---------------------------------------------------------------------------

class TestReportRewards:

    def test_correct_label_reward(self):
        state = _report_state("lab", submitted="lab", classified=True)
        a = _action(ActionType.CLASSIFY_REPORT, report_id="R1", label="lab")
        r = engine.compute(a, {"success": True, "correct": True}, state)
        assert r == 1.0

    def test_wrong_label_penalty(self):
        state = _report_state("lab", submitted="imaging", classified=True)
        a = _action(ActionType.CLASSIFY_REPORT, report_id="R1", label="imaging")
        r = engine.compute(a, {"success": True, "correct": False}, state)
        assert r == -0.5

    def test_invalid_label_penalty(self):
        state = _report_state("lab")
        a = _action(ActionType.CLASSIFY_REPORT, report_id="R1", label="garbage")
        r = engine.compute(a, {"success": False, "correct": False}, state)
        assert r == -0.1


# ---------------------------------------------------------------------------
# Billing rewards
# ---------------------------------------------------------------------------

class TestBillingRewards:

    def test_validate_correct(self):
        state = _billing_state(
            code_validated=True,
            code_correct_result=True,  # matches code_is_valid=True
        )
        a = _action(ActionType.VALIDATE_BILLING_CODE, claim_id="C1", code="85025")
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.2 + 0.0833, abs=1e-3)

    def test_validate_wrong_result(self):
        # Agent said invalid, but code IS valid → partial credit only
        state = _billing_state(
            code_validated=True,
            code_correct_result=False,  # mismatch with code_is_valid=True
        )
        a = _action(ActionType.VALIDATE_BILLING_CODE, claim_id="C1", code="85025")
        r = engine.compute(a, {"success": True}, state)
        assert r == 0.05

    def test_flag_real_discrepancy(self):
        state = _billing_state(has_discrepancy=True, discrepancy_flagged=True)
        a = _action(ActionType.FLAG_DISCREPANCY, claim_id="C1", reason="mismatch")
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.3 + 0.0833, abs=1e-3)

    def test_false_flag_penalty(self):
        state = _billing_state(has_discrepancy=False, discrepancy_flagged=True)
        a = _action(ActionType.FLAG_DISCREPANCY, claim_id="C1", reason="false alarm")
        r = engine.compute(a, {"success": True}, state)
        assert r == -0.2

    def test_correct_approve(self):
        state = _billing_state(
            code_validated=True,
            correct_final_decision="approve_claim",
            final_decision="approve_claim",
        )
        a = _action(ActionType.APPROVE_CLAIM, claim_id="C1")
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.5 + 0.0833, abs=1e-3)

    def test_wrong_approve(self):
        state = _billing_state(
            code_validated=True,
            correct_final_decision="reject_claim",
            final_decision="approve_claim",
        )
        a = _action(ActionType.APPROVE_CLAIM, claim_id="C1")
        r = engine.compute(a, {"success": True}, state)
        assert r == -0.5

    def test_approve_without_validation(self):
        state = _billing_state(code_validated=False)
        a = _action(ActionType.APPROVE_CLAIM, claim_id="C1")
        r = engine.compute(a, {"success": False}, state)
        assert r == -0.2

    def test_correct_reject(self):
        state = _billing_state(
            correct_final_decision="reject_claim",
            final_decision="reject_claim",
        )
        a = _action(ActionType.REJECT_CLAIM, claim_id="C1", reason="not covered")
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.5 + 0.0833, abs=1e-3)

    def test_wrong_reject(self):
        state = _billing_state(
            correct_final_decision="approve_claim",
            final_decision="reject_claim",
        )
        a = _action(ActionType.REJECT_CLAIM, claim_id="C1", reason="oops")
        r = engine.compute(a, {"success": True}, state)
        assert r == -0.5


# ---------------------------------------------------------------------------
# Blood bank rewards
# ---------------------------------------------------------------------------

class TestBloodBankRewards:

    def test_full_correct_allocation(self):
        state = _blood_state(
            allocated_type="A+", allocated_units=2,
            allocation_fulfilled=True,
        )
        a = _action(ActionType.ALLOCATE_BLOOD,
                    request_id="B1", blood_type="A+", units=2)
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.8 + 0.0833, abs=1e-3)

    def test_partial_correct_allocation(self):
        state = _blood_state(
            allocated_type="A+", allocated_units=1,  # only 1 of 2
            allocation_fulfilled=False,
        )
        a = _action(ActionType.ALLOCATE_BLOOD,
                    request_id="B1", blood_type="A+", units=1)
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.6 + 0.0833, abs=1e-3)

    def test_wrong_type_allocated(self):
        state = _blood_state(
            allocated_type="B+", allocated_units=2,
            allocation_fulfilled=True,
        )
        a = _action(ActionType.ALLOCATE_BLOOD,
                    request_id="B1", blood_type="B+", units=2)
        r = engine.compute(a, {"success": True}, state)
        assert r == -0.4

    def test_failed_allocation(self):
        state = _blood_state()
        a = _action(ActionType.ALLOCATE_BLOOD,
                    request_id="B1", blood_type="A+", units=999)
        r = engine.compute(a, {"success": False}, state)
        assert r == -0.3

    def test_needed_restock(self):
        state = _blood_state(should_request_restock=True, restock_requested=True)
        a = _action(ActionType.REQUEST_RESTOCK, blood_type="A+", units_requested=5)
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.2 + 0.0833, abs=1e-3)

    def test_unnecessary_restock(self):
        state = _blood_state(should_request_restock=False)
        a = _action(ActionType.REQUEST_RESTOCK, blood_type="A+", units_requested=5)
        r = engine.compute(a, {"success": True}, state)
        assert r == 0.0

    def test_correct_discard(self):
        state = _blood_state(should_discard_types=["B+"], units_discarded=2)
        a = _action(ActionType.DISCARD_EXPIRED,
                    blood_type="B+", units_to_discard=2)
        r = engine.compute(a, {"success": True}, state)
        assert r == pytest.approx(0.2 + 0.0833, abs=1e-3)

    def test_wasteful_discard(self):
        state = _blood_state(should_discard_types=[], units_discarded=2)
        a = _action(ActionType.DISCARD_EXPIRED,
                    blood_type="A+", units_to_discard=2)
        r = engine.compute(a, {"success": True}, state)
        assert r == -0.2
