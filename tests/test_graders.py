"""
tests/test_graders.py
=====================
Unit tests for GraderEngine.
Verifies that graders return different scores for different outcomes
and that partial credit is awarded correctly.

Run with: pytest tests/test_graders.py -v
"""

import pytest
from app.graders import GraderEngine
from app.models import (
    BillingState,
    BloodBankState,
    Difficulty,
    EpisodeStatus,
    InternalState,
    ReportState,
    ScenarioDefinition,
    TaskType,
)
from app.scenarios import ScenarioLoader

grader = GraderEngine()
loader = ScenarioLoader("scenarios")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_report_state(submitted: str, correct: str, steps: int) -> InternalState:
    return InternalState(
        episode_id="test-ep",
        scenario_id="report_easy",
        task_type=TaskType.REPORT_CLASSIFICATION,
        difficulty=Difficulty.EASY,
        step_number=steps,
        max_steps=3,
        report_state=ReportState(
            report_id="RPT-001",
            correct_label=correct,
            submitted_label=submitted,
            classified=True,
        ),
    )


def make_billing_state(**kwargs) -> BillingState:
    defaults = dict(
        claim_id="CLM-101",
        correct_code="85025",
        code_is_valid=True,
        insurance_covers=True,
        has_discrepancy=False,
        correct_final_decision="approve_claim",
        code_validated=False,
        code_correct_result=None,
        insurance_verified=False,
        insurance_correct_result=None,
        discrepancy_flagged=False,
        final_decision=None,
    )
    defaults.update(kwargs)
    return BillingState(**defaults)


def make_billing_internal(bs: BillingState, scenario_id="billing_easy") -> InternalState:
    return InternalState(
        episode_id="test-ep",
        scenario_id=scenario_id,
        task_type=TaskType.BILLING_VERIFICATION,
        difficulty=Difficulty.EASY,
        step_number=3,
        max_steps=6,
        billing_state=bs,
    )


def make_blood_state(**kwargs) -> BloodBankState:
    defaults = dict(
        request_id="BLD-001",
        requested_type="A+",
        requested_units=2,
        urgency="routine",
        inventory={"A+": 10, "A-": 3, "B+": 7, "B-": 1,
                   "O+": 12, "O-": 6, "AB+": 3, "AB-": 1},
        initial_inventory={"A+": 12, "A-": 4, "B+": 8, "B-": 2,
                           "O+": 15, "O-": 6, "AB+": 3, "AB-": 1},
        expiry_data={},
        should_discard_types=[],
        should_request_restock=False,
        correct_allocation_type="A+",
        restock_requested=False,
        allocation_fulfilled=True,
        allocated_type="A+",
        allocated_units=2,
        units_discarded=0,
    )
    defaults.update(kwargs)
    return BloodBankState(**defaults)


def make_blood_internal(bb: BloodBankState) -> InternalState:
    return InternalState(
        episode_id="test-ep",
        scenario_id="bloodbank_easy",
        task_type=TaskType.BLOOD_BANK,
        difficulty=Difficulty.EASY,
        step_number=1,
        max_steps=5,
        blood_bank_state=bb,
    )


# ---------------------------------------------------------------------------
# Report Classification Grader
# ---------------------------------------------------------------------------


class TestReportGrader:

    def test_perfect_score(self):
        scenario = loader.load("report_easy")
        state = make_report_state("lab", "lab", steps=1)
        score = grader.grade(state, scenario)
        assert score == 1.0

    def test_correct_label_slow(self):
        scenario = loader.load("report_easy")
        state = make_report_state("lab", "lab", steps=3)
        score = grader.grade(state, scenario)
        assert score == 0.8   # correct label, no efficiency bonus (steps=3)

    def test_wrong_label(self):
        scenario = loader.load("report_easy")
        state = make_report_state("imaging", "lab", steps=1)
        score = grader.grade(state, scenario)
        assert score == 0.0   # wrong label → no correctness, no efficiency bonus

    def test_not_classified(self):
        scenario = loader.load("report_easy")
        state = InternalState(
            episode_id="test-ep",
            scenario_id="report_easy",
            task_type=TaskType.REPORT_CLASSIFICATION,
            difficulty=Difficulty.EASY,
            step_number=3,
            max_steps=3,
            report_state=ReportState(
                report_id="RPT-001",
                correct_label="lab",
                classified=False,
            ),
        )
        score = grader.grade(state, scenario)
        assert score == 0.0

    def test_scores_are_not_all_identical(self):
        scenario = loader.load("report_easy")
        scores = {
            grader.grade(make_report_state("lab", "lab", 1), scenario),
            grader.grade(make_report_state("lab", "lab", 3), scenario),
            grader.grade(make_report_state("imaging", "lab", 1), scenario),
        }
        assert len(scores) > 1   # not all the same


# ---------------------------------------------------------------------------
# Billing Verification Grader
# ---------------------------------------------------------------------------


class TestBillingGrader:

    def test_perfect_score(self):
        scenario = loader.load("billing_easy")
        bs = make_billing_state(
            code_validated=True,
            code_correct_result=True,
            insurance_verified=True,
            insurance_correct_result=True,
            final_decision="approve_claim",
        )
        score = grader.grade(make_billing_internal(bs), scenario)
        assert score == 1.0

    def test_zero_score_no_actions(self):
        scenario = loader.load("billing_easy")
        bs = make_billing_state()
        score = grader.grade(make_billing_internal(bs), scenario)
        # No validation, no insurance check, no discrepancy action, no decision
        # Only the "no discrepancy" bonus applies (0.20) since has_discrepancy=False
        assert score == 0.20

    def test_partial_credit_code_only(self):
        scenario = loader.load("billing_easy")
        bs = make_billing_state(
            code_validated=True,
            code_correct_result=True,
        )
        score = grader.grade(make_billing_internal(bs), scenario)
        # 0.10 (validated) + 0.15 (correct) + 0.20 (no discrepancy) = 0.45
        assert score == pytest.approx(0.45)

    def test_wrong_final_decision_penalised(self):
        scenario = loader.load("billing_easy")
        bs = make_billing_state(
            code_validated=True,
            code_correct_result=True,
            insurance_verified=True,
            insurance_correct_result=True,
            final_decision="reject_claim",   # wrong
        )
        score = grader.grade(make_billing_internal(bs), scenario)
        # 0.10 (validated) + 0.15 (correct code eval) + 0.20 (correct ins eval)
        # + 0.20 (no discrepancy, not falsely flagged) + 0.00 (wrong decision) = 0.65
        assert score == pytest.approx(0.65)

    def test_hard_scenario_with_discrepancy(self):
        scenario = loader.load("billing_hard")
        bs = make_billing_state(
            claim_id="CLM-303",
            correct_code="FAKE99",
            code_is_valid=False,
            insurance_covers=False,
            has_discrepancy=True,
            correct_final_decision="reject_claim",
            code_validated=True,
            code_correct_result=True,
            insurance_verified=True,
            insurance_correct_result=True,
            discrepancy_flagged=True,
            final_decision="reject_claim",
        )
        state = make_billing_internal(bs, "billing_hard")
        score = grader.grade(state, scenario)
        assert score == 1.0

    def test_missed_discrepancy_no_points(self):
        scenario = loader.load("billing_hard")
        bs = make_billing_state(
            claim_id="CLM-303",
            correct_code="FAKE99",
            code_is_valid=False,
            insurance_covers=False,
            has_discrepancy=True,
            correct_final_decision="reject_claim",
            code_validated=True,
            code_correct_result=True,
            insurance_verified=True,
            insurance_correct_result=True,
            discrepancy_flagged=False,   # missed
            final_decision="reject_claim",
        )
        state = make_billing_internal(bs, "billing_hard")
        score = grader.grade(state, scenario)
        # 0.10 + 0.15 + 0.20 + 0.00 (missed discrepancy) + 0.35 = 0.80
        assert score == pytest.approx(0.80, abs=1e-3)

    def test_scores_vary_across_outcomes(self):
        scenario = loader.load("billing_easy")
        perfect_bs = make_billing_state(
            code_validated=True, code_correct_result=True,
            insurance_verified=True, insurance_correct_result=True,
            final_decision="approve_claim",
        )
        empty_bs = make_billing_state()
        wrong_bs = make_billing_state(
            code_validated=True, code_correct_result=True,
            insurance_verified=True, insurance_correct_result=True,
            final_decision="reject_claim",
        )
        s1 = grader.grade(make_billing_internal(perfect_bs), scenario)
        s2 = grader.grade(make_billing_internal(empty_bs), scenario)
        s3 = grader.grade(make_billing_internal(wrong_bs), scenario)
        assert len({s1, s2, s3}) == 3  # all distinct


# ---------------------------------------------------------------------------
# Blood Bank Grader
# ---------------------------------------------------------------------------


class TestBloodBankGrader:

    def test_perfect_score_easy(self):
        scenario = loader.load("bloodbank_easy")
        bb = make_blood_state()
        score = grader.grade(make_blood_internal(bb), scenario)
        # type(0.35) + units(0.15) + expiry_clean(0.20) + no_waste(0.15) + restock_ok(0.15) = 1.0
        assert score == 1.0

    def test_wrong_blood_type_penalised(self):
        scenario = loader.load("bloodbank_easy")
        bb = make_blood_state(allocated_type="B+", allocated_units=2)
        score = grader.grade(make_blood_internal(bb), scenario)
        # No type points, no unit points; expiry(0.20) + no_waste(0.15) + restock(0.15) = 0.50
        assert score == pytest.approx(0.50)

    def test_partial_correct_type_wrong_units(self):
        scenario = loader.load("bloodbank_easy")
        bb = make_blood_state(allocated_type="A+", allocated_units=1)  # 1 of 2 requested
        score = grader.grade(make_blood_internal(bb), scenario)
        # Type(0.35) + no units bonus + expiry(0.20) + no_waste(0.15) + restock(0.15) = 0.85
        assert score == pytest.approx(0.85)

    def test_unnecessary_discard_penalised(self):
        scenario = loader.load("bloodbank_easy")  # should_discard_types = []
        bb = make_blood_state(units_discarded=3)
        score = grader.grade(make_blood_internal(bb), scenario)
        # type(0.35) + units(0.15) + expiry: discard NOT needed but done → 0.00
        # + no_waste penalty: −0.15 applied
        # + restock(0.15) = 0.50
        assert score == pytest.approx(0.50)

    def test_hard_scenario_full_correct(self):
        scenario = loader.load("bloodbank_hard")
        bb = make_blood_state(
            request_id="BLD-099",
            requested_type="O-",
            requested_units=2,
            urgency="emergency",
            inventory={"A+": 7, "A-": 3, "B+": 7, "B-": 2,
                       "O+": 12, "O-": 0, "AB+": 5, "AB-": 0},
            initial_inventory={"A+": 10, "A-": 3, "B+": 7, "B-": 2,
                               "O+": 12, "O-": 1, "AB+": 5, "AB-": 0},
            expiry_data={"A+": 3},
            should_discard_types=["A+"],
            should_request_restock=True,
            correct_allocation_type="O-",
            allocated_type="O-",
            allocated_units=1,
            units_discarded=3,
            restock_requested=True,
            allocation_fulfilled=True,
        )
        state = InternalState(
            episode_id="test-ep",
            scenario_id="bloodbank_hard",
            task_type=TaskType.BLOOD_BANK,
            difficulty=Difficulty.HARD,
            step_number=3,
            max_steps=10,
            blood_bank_state=bb,
        )
        score = grader.grade(state, scenario)
        # type(0.35) + no full units(0) + expiry_handled(0.20+0.15) + restock(0.15) = 0.85
        assert score >= 0.85

    def test_scores_not_all_identical(self):
        scenario = loader.load("bloodbank_easy")
        s1 = grader.grade(make_blood_internal(make_blood_state()), scenario)
        s2 = grader.grade(make_blood_internal(make_blood_state(allocated_type="B+")), scenario)
        s3 = grader.grade(make_blood_internal(make_blood_state(allocated_units=0,
                                                                allocated_type=None,
                                                                allocation_fulfilled=False)), scenario)
        assert len({s1, s2, s3}) > 1
