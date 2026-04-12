"""
app/graders.py
==============
GraderEngine – deterministic episode-end scoring.

Called exactly once per episode when done=True.
Returns a float in [0.0, 1.0] representing overall episode quality.

Design principles
-----------------
* Purely deterministic: same action_history + ground_truth → same score.
* Never returns the same score regardless of what the agent did
  (verified via unit tests with all-correct and all-wrong inputs).
* Partial credit: doing some steps correctly is better than doing nothing.
* Destructive mistakes (wrong final decision, incompatible blood) reduce score.
"""

from __future__ import annotations

from app.models import InternalState, ScenarioDefinition, TaskType
from app.utils import clamp


class GraderEngine:
    """Grade a completed episode. Returns score in [0.0, 1.0]."""

    def grade(
        self,
        state: InternalState,
        scenario: ScenarioDefinition,
    ) -> float:
        gt = scenario.ground_truth
        if state.task_type == TaskType.REPORT_CLASSIFICATION:
            return max(0.001, min(0.999, self._grade_report(state, gt)))
        if state.task_type == TaskType.BILLING_VERIFICATION:
            return max(0.001, min(0.999, self._grade_billing(state, gt)))
        if state.task_type == TaskType.BLOOD_BANK:
            return max(0.001, min(0.999, self._grade_blood_bank(state, gt)))
        if state.task_type == TaskType.ICU_BED_SCHEDULING:
            return max(0.001, min(0.999, self._grade_icu(state, gt)))
        return 0.00101

    # ------------------------------------------------------------------
    # Task 1 – Report Classification
    # ------------------------------------------------------------------
    # Max score breakdown:
    #   Correct label submitted          : 0.80
    #   Efficiency (≤1 step)             : 0.20
    # ------------------------------------------------------------------

    def _grade_report(self, state: InternalState, gt: dict) -> float:
        rs = state.report_state
        score = 0.0

        correct = rs.classified and rs.submitted_label == gt["correct_label"]

        # Correctness (80%)
        if correct:
            score += 0.80

        # Efficiency (20%) — only awarded when classification is correct
        # Penalising inefficiency on a wrong answer would double-punish the agent.
        if correct:
            steps = state.step_number
            if steps <= 1:
                score += 0.20
            elif steps == 2:
                score += 0.10
            # >2 steps → 0 efficiency bonus (agent looped before classifying)

        return round(min(0.998, max(0.002, score)), 3)

    # ------------------------------------------------------------------
    # Task 2 – Billing & Insurance Verification
    # ------------------------------------------------------------------
    # Max score breakdown:
    #   Code validation attempted        : 0.10
    #   Code validation result correct   : 0.15
    #   Insurance verification correct   : 0.20
    #   Discrepancy handled correctly    : 0.20
    #   Correct final decision           : 0.35
    # ------------------------------------------------------------------

    def _grade_billing(self, state: InternalState, gt: dict) -> float:
        bs = state.billing_state
        score = 0.0

        # Code validation attempted (0.10)
        if bs.code_validated:
            score += 0.10
            # Code evaluation correct (0.15)
            if bs.code_correct_result == gt["code_is_valid"]:
                score += 0.15

        # Insurance verification correct (0.20)
        if bs.insurance_verified:
            if bs.insurance_correct_result == gt["insurance_covers"]:
                score += 0.20

        # Discrepancy handling (0.20)
        if gt["has_discrepancy"]:
            if bs.discrepancy_flagged:
                score += 0.20
            # else: missed discrepancy → no points
        else:
            # No discrepancy existed; full marks whether or not agent flagged
            # UNLESS agent falsely flagged (penalise false positives)
            if not bs.discrepancy_flagged:
                score += 0.20
            else:
                score += 0.05  # false flag – partial credit only

        # Correct final decision (0.35)
        if bs.final_decision == gt["correct_final_decision"]:
            score += 0.35
        # Wrong decision or no decision → 0 points here

        return round(min(0.998, max(0.002, score)), 3)

    # ------------------------------------------------------------------
    # Task 3 – Blood Bank Management
    # ------------------------------------------------------------------
    # Max score breakdown:
    #   Correct blood type allocated     : 0.35
    #   Correct units fulfilled          : 0.15
    #   Expiry handling correct          : 0.20
    #   Restock handling correct         : 0.15
    #   No wasteful discards             : 0.15
    # ------------------------------------------------------------------

    def _grade_blood_bank(self, state: InternalState, gt: dict) -> float:
        bb = state.blood_bank_state
        score = 0.0

        # Correct allocation type (0.35)
        correct_types = {gt["correct_allocation_type"], "O-"}
        type_correct = bb.allocated_type in correct_types

        if type_correct:
            score += 0.35

            # Correct units (0.15) — only meaningful when type is right
            # max_possible: we can't allocate more than what was initially available
            max_possible = min(
                bb.requested_units,
                bb.initial_inventory.get(bb.allocated_type, 0),
            )
            if max_possible > 0 and bb.allocated_units >= max_possible:
                score += 0.15

        # Expiry / inventory-hygiene handling (up to 0.35 total)
        should_discard = gt.get("should_discard_types", [])
        if should_discard:
            if bb.units_discarded > 0:
                score += 0.20   # correctly discarded expiring units
                score += 0.15   # clean-hands: did it properly
        else:
            if bb.units_discarded == 0:
                score += 0.20   # correctly did not discard
                score += 0.15   # clean-hands: no wasteful discards
            else:
                # Wasteful discard: lose both sub-components + apply penalty
                score = max(0.001, score - 0.15)

        # Restock handling (0.15)
        if gt.get("should_request_restock", False):
            if bb.restock_requested:
                score += 0.15
        else:
            score += 0.15

        return round(min(0.998, max(0.002, score)), 3)

    # ------------------------------------------------------------------
    # Task 4 – ICU Bed Scheduling
    # ------------------------------------------------------------------
    # Max score breakdown:
    #   Assessed the highest-priority patient   : 0.15
    #   Discharged the correct step-down patient: 0.15  (if required)
    #   Escalated issue when needed             : 0.15  (if required)
    #   Assigned the correct bed                : 0.25
    #   Correct patient admitted                : 0.20
    #   Admission confirmed                     : 0.10
    # ------------------------------------------------------------------

    def _grade_icu(self, state: InternalState, gt: dict) -> float:
        icu = state.icu_state
        score = 0.0

        correct_patient = gt["correct_patient_to_admit"]
        correct_bed     = gt["correct_bed_id"]
        need_discharge  = gt.get("should_discharge_patient")
        need_escalate   = gt.get("should_escalate", False)

        # Assessed the right patient (0.15)
        if icu.patient_assessed and icu.assessed_patient_id == correct_patient:
            score += 0.15
        elif icu.patient_assessed:
            score += 0.05  # assessed someone but wrong patient

        # Discharge handling (0.15)
        if need_discharge:
            if icu.patient_discharged and icu.discharged_patient_id == need_discharge:
                score += 0.15
            elif icu.patient_discharged:
                score += 0.05  # discharged wrong patient
            # else: did not discharge when required → 0
        else:
            # No discharge needed
            if not icu.patient_discharged:
                score += 0.15  # correctly did NOT discharge
            else:
                score += 0.001  # unnecessary discharge – slight penalty handled in rewards

        # Escalation handling (0.15)
        if need_escalate:
            if icu.issue_escalated:
                score += 0.15
            # else: failed to escalate → 0
        else:
            if not icu.issue_escalated:
                score += 0.15  # correctly did not escalate
            else:
                score += 0.05  # false escalation – partial only

        # Correct bed assigned (0.25)
        if icu.bed_assigned and icu.assigned_bed_id == correct_bed:
            score += 0.25
        elif icu.bed_assigned:
            score += 0.05  # assigned a bed but wrong one

        # Correct patient admitted (0.20)
        if icu.bed_assigned and icu.assigned_patient_id == correct_patient:
            score += 0.20
        elif icu.bed_assigned:
            score += 0.05  # admitted wrong patient

        # Admission confirmed (0.10)
        if icu.admission_confirmed:
            score += 0.10

        return round(min(0.998, max(0.002, score)), 3)
