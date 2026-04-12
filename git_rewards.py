"""
app/rewards.py
==============
RewardEngine – pure, stateless step-level reward computation.

All reward logic lives here. No side effects; no state mutation.
Called once per step() after the task handler has updated state.

Reward philosophy
-----------------
* Dense signals: the agent gets feedback at every step.
* Correct procedural steps earn partial credit even if the episode ends badly.
* Wrong final decisions are penalised heavily.
* Unnecessary actions (false flags, useless discards) cost a small penalty.
* Looping (repeating the same action) gets a fixed -0.05 deduction
  (handled in env.py before calling this module).
* Invalid actions get -0.1 (also handled in env.py).

Time-decay (blood bank & billing)
----------------------------------
For time-sensitive tasks (blood_bank, billing_verification) correct actions
earn a small bonus that decreases with each step taken. This encourages the
agent to reach the correct decision efficiently without unnecessary detours.

  decay_bonus = BASE_DECAY * max(0, 1 - step / MAX_DECAY_STEPS)

  BASE_DECAY      = 0.10   (max extra reward for acting at step 0)
  MAX_DECAY_STEPS = 6      (bonus reaches 0 by step 6)

The decay bonus is added on top of the base action reward, so it never
converts a negative reward into a positive one.
"""

from __future__ import annotations

from app.models import Action, ActionType, InternalState, TaskType

BASE_DECAY: float = 0.10
MAX_DECAY_STEPS: int = 6


def _time_decay_bonus(step_number: int) -> float:
    """Return a small positive bonus that shrinks linearly to 0 by step 6."""
    return round(BASE_DECAY * max(0.0, 1.0 - step_number / MAX_DECAY_STEPS), 4)


class RewardEngine:
    """
    Compute the step reward for a completed action.

    Parameters
    ----------
    action  : the action that was just taken
    outcome : dict returned by the task handler
    state   : InternalState *after* the task handler has updated it
    """

    def compute(
        self,
        action: Action,
        outcome: dict,
        state: InternalState,
    ) -> float:
        if state.task_type == TaskType.REPORT_CLASSIFICATION:
            return self._reward_report(action, outcome, state)
        if state.task_type == TaskType.BILLING_VERIFICATION:
            return self._reward_billing(action, outcome, state)
        if state.task_type == TaskType.BLOOD_BANK:
            return self._reward_blood_bank(action, outcome, state)
        if state.task_type == TaskType.ICU_BED_SCHEDULING:
            return self._reward_icu(action, outcome, state)
        return 0.0

    # ------------------------------------------------------------------
    # Task 1 – Report Classification
    # ------------------------------------------------------------------

    def _reward_report(
        self,
        action: Action,
        outcome: dict,
        state: InternalState,
    ) -> float:
        if not outcome.get("success"):
            return -0.1  # invalid label supplied

        if outcome.get("correct"):
            return 1.0   # correct label

        return -0.5      # classified but wrong label

    # ------------------------------------------------------------------
    # Task 2 – Billing & Insurance Verification  (with time-decay)
    # ------------------------------------------------------------------

    def _reward_billing(
        self,
        action: Action,
        outcome: dict,
        state: InternalState,
    ) -> float:
        at = action.action_type
        bs = state.billing_state
        step = state.step_number
        decay = _time_decay_bonus(step)

        # ---- validate_billing_code ----------------------------------------
        if at == ActionType.VALIDATE_BILLING_CODE:
            if not outcome.get("success"):
                return -0.1
            correct = bs.code_correct_result == bs.code_is_valid
            base = 0.2 if correct else 0.05
            return base + (decay if correct else 0.0)

        # ---- verify_insurance ---------------------------------------------
        if at == ActionType.VERIFY_INSURANCE:
            if not outcome.get("success"):
                return -0.1
            correct = bs.insurance_correct_result == bs.insurance_covers
            base = 0.2 if correct else 0.05
            return base + (decay if correct else 0.0)

        # ---- flag_discrepancy ---------------------------------------------
        if at == ActionType.FLAG_DISCREPANCY:
            if not outcome.get("success"):
                return -0.1
            if bs.has_discrepancy:
                return 0.3 + decay   # correctly identified a real discrepancy
            return -0.2              # false positive – penalty (no decay bonus)

        # ---- approve_claim ------------------------------------------------
        if at == ActionType.APPROVE_CLAIM:
            if not outcome.get("success"):
                return -0.2   # tried to approve without validating code
            if bs.correct_final_decision == "approve_claim":
                return 0.5 + decay   # correct decision
            return -0.5              # wrong decision (should have rejected)

        # ---- reject_claim -------------------------------------------------
        if at == ActionType.REJECT_CLAIM:
            if bs.correct_final_decision == "reject_claim":
                return 0.5 + decay   # correct decision
            return -0.5              # wrong decision (should have approved)

        return 0.0

    # ------------------------------------------------------------------
    # Task 3 – Blood Bank Management  (with time-decay)
    # ------------------------------------------------------------------

    def _reward_blood_bank(
        self,
        action: Action,
        outcome: dict,
        state: InternalState,
    ) -> float:
        at = action.action_type
        bb = state.blood_bank_state
        step = state.step_number
        decay = _time_decay_bonus(step)

        # ---- allocate_blood / use_compatible_type -------------------------
        if at in (ActionType.ALLOCATE_BLOOD, ActionType.USE_COMPATIBLE_TYPE):
            if not outcome.get("success"):
                return -0.3  # tried to allocate unavailable or incompatible

            correct_types = {bb.correct_allocation_type, "O-"}
            if bb.allocated_type in correct_types:
                if bb.allocated_units >= bb.requested_units:
                    return 0.8 + decay   # full correct allocation
                return 0.6 + decay       # partial correct allocation
            else:
                return -0.4              # wrong blood type (no decay bonus)

        # ---- request_restock -----------------------------------------------
        if at == ActionType.REQUEST_RESTOCK:
            if not outcome.get("success"):
                return -0.1
            if bb.should_request_restock:
                return 0.2 + decay   # needed and requested
            return 0.0               # not needed, but not harmful

        # ---- discard_expired -----------------------------------------------
        if at == ActionType.DISCARD_EXPIRED:
            if not outcome.get("success"):
                return -0.1
            btype = action.payload.get("blood_type", "")
            if btype in bb.should_discard_types:
                return 0.2 + decay   # correctly discarding an expiring type
            return -0.2              # discarding non-expiring units – wasteful

        return 0.0

    # ------------------------------------------------------------------
    # Task 4 – ICU Bed Scheduling
    # ------------------------------------------------------------------

    def _reward_icu(
        self,
        action: Action,
        outcome: dict,
        state: InternalState,
    ) -> float:
        at = action.action_type
        icu = state.icu_state
        gt_patient  = icu.correct_patient_to_admit
        gt_bed      = icu.correct_bed_id
        need_discharge = icu.should_discharge_patient
        need_escalate  = icu.should_escalate

        if not outcome.get("success"):
            return -0.1  # generic invalid-action penalty

        # ---- assess_patient ------------------------------------------------
        if at == ActionType.ASSESS_PATIENT:
            if icu.assessed_patient_id == gt_patient:
                return 0.3   # assessed the right patient
            return -0.1      # assessed but wrong patient

        # ---- discharge_patient ---------------------------------------------
        if at == ActionType.DISCHARGE_PATIENT:
            if need_discharge and icu.discharged_patient_id == need_discharge:
                return 0.3   # correctly freed the step-down bed
            if need_discharge and icu.discharged_patient_id != need_discharge:
                return -0.2  # discharged wrong patient
            # No discharge was required
            return -0.2      # unnecessary discharge

        # ---- escalate_issue ------------------------------------------------
        if at == ActionType.ESCALATE_ISSUE:
            if need_escalate:
                return 0.3   # correctly escalated a real issue
            return -0.2      # false escalation

        # ---- assign_bed ----------------------------------------------------
        if at == ActionType.ASSIGN_BED:
            correct_bed_assigned     = icu.assigned_bed_id == gt_bed
            correct_patient_assigned = icu.assigned_patient_id == gt_patient
            if correct_bed_assigned and correct_patient_assigned:
                return 0.5   # perfect assignment
            if correct_patient_assigned:
                return 0.2   # right patient, wrong bed
            if correct_bed_assigned:
                return 0.1   # right bed, wrong patient
            return -0.3      # both wrong

        # ---- confirm_admission ---------------------------------------------
        if at == ActionType.CONFIRM_ADMISSION:
            # Reward quality of the overall decision at closure
            correct_patient = icu.assigned_patient_id == gt_patient
            correct_bed     = icu.assigned_bed_id == gt_bed
            missed_escalate = need_escalate and not icu.issue_escalated
            missed_discharge = need_discharge and not icu.patient_discharged
            if correct_patient and correct_bed and not missed_escalate and not missed_discharge:
                return 1.0   # perfect episode closure
            if correct_patient and correct_bed:
                return 0.5   # right patient and bed but missed auxiliary steps
            return 0.1       # confirmed but suboptimal

        return 0.0
