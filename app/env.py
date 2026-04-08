"""
app/env.py
==========
HospitalOpsEnv – main OpenEnv-compatible environment class.

Public API
----------
  env = HospitalOpsEnv()
  obs              = env.reset(scenario_id)
  obs, r, done, info = env.step(action)
  state            = env.state()

All return types are Pydantic models (see app/models.py).
"""

from __future__ import annotations

import uuid
from typing import Optional, Tuple

from app.graders import GraderEngine
from app.models import (
    Action,
    ActionType,
    BillingState,
    BloodBankState,
    Difficulty,
    EpisodeStatus,
    IcuSchedulingState,
    InternalState,
    Observation,
    ReportState,
    ScenarioDefinition,
    StepInfo,
    TaskType,
)
from app.rewards import RewardEngine
from app.scenarios import ScenarioLoader
from app.tasks import (
    BillingVerificationTask,
    BloodBankTask,
    IcuSchedulingTask,
    ReportClassificationTask,
)

# ---------------------------------------------------------------------------
# Which action types are valid for each task
# ---------------------------------------------------------------------------

TASK_ALLOWED_ACTIONS: dict[TaskType, list[str]] = {
    TaskType.REPORT_CLASSIFICATION: [
        "classify_report",
    ],
    TaskType.BILLING_VERIFICATION: [
        "validate_billing_code",
        "verify_insurance",
        "flag_discrepancy",
        "approve_claim",
        "reject_claim",
    ],
    TaskType.BLOOD_BANK: [
        "allocate_blood",
        "request_restock",
        "discard_expired",
        "use_compatible_type",
    ],
    TaskType.ICU_BED_SCHEDULING: [
        "assess_patient",
        "assign_bed",
        "discharge_patient",
        "confirm_admission",
        "escalate_issue",
    ],
}

# Required payload keys per action type
REQUIRED_PAYLOAD_KEYS: dict[str, list[str]] = {
    "classify_report":        ["report_id", "label"],
    "validate_billing_code":  ["claim_id", "code"],
    "verify_insurance":       ["claim_id", "plan_id"],
    "flag_discrepancy":       ["claim_id", "reason"],
    "approve_claim":          ["claim_id"],
    "reject_claim":           ["claim_id", "reason"],
    "allocate_blood":         ["request_id", "blood_type", "units"],
    "request_restock":        ["blood_type", "units_requested"],
    "discard_expired":        ["blood_type", "units_to_discard"],
    "use_compatible_type":    ["request_id", "substitute_type", "units"],
    "assess_patient":         ["patient_id"],
    "assign_bed":             ["patient_id", "bed_id"],
    "discharge_patient":      ["patient_id"],
    "confirm_admission":      ["patient_id"],
    "escalate_issue":         ["reason"],
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------


class HospitalOpsEnv:
    """
    Rule-based, deterministic OpenEnv environment simulating hospital
    operational workflows across four tasks.

    Parameters
    ----------
    scenarios_dir : path to the directory containing scenario JSON files
                    (default: 'scenarios', relative to working directory)
    """

    def __init__(self, scenarios_dir: str = "scenarios") -> None:
        self._loader = ScenarioLoader(scenarios_dir)
        self._reward_engine = RewardEngine()
        self._grader_engine = GraderEngine()
        self._state: Optional[InternalState] = None
        self._scenario: Optional[ScenarioDefinition] = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, scenario_id: str) -> Observation:
        """
        Load *scenario_id*, initialise internal state, return first observation.

        Parameters
        ----------
        scenario_id : one of the IDs listed in openenv.yaml (e.g. 'billing_medium')

        Returns
        -------
        Observation  – initial visible state for the agent
        """
        self._scenario = self._loader.load(scenario_id)
        episode_id = f"ep-{uuid.uuid4().hex[:10]}"
        self._state = self._build_initial_state(self._scenario, episode_id)
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, StepInfo]:
        """
        Apply *action* to the current state.

        Parameters
        ----------
        action : Action  – must have episode_id matching the active episode

        Returns
        -------
        (Observation, float, bool, StepInfo)
          obs    – updated visible observation
          reward – step reward (may be negative)
          done   – True when the episode has ended
          info   – StepInfo with outcome text; grader_score when done=True
        """
        if self._state is None or self._scenario is None:
            raise RuntimeError("Call reset(scenario_id) before step().")

        if self._state.status != EpisodeStatus.RUNNING:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        # ---- episode_id check ---------------------------------------------
        if action.episode_id != self._state.episode_id:
            return (
                self._build_observation(),
                -0.1,
                False,
                StepInfo(
                    action_valid=False,
                    validation_errors=["episode_id mismatch"],
                    outcome="episode_id in action does not match active episode.",
                ),
            )

        # ---- action type allowed for this task? ---------------------------
        allowed = TASK_ALLOWED_ACTIONS[self._state.task_type]
        if action.action_type.value not in allowed:
            return (
                self._build_observation(),
                -0.1,
                False,
                StepInfo(
                    action_valid=False,
                    validation_errors=[
                        f"Action '{action.action_type.value}' is not allowed "
                        f"for task '{self._state.task_type.value}'. "
                        f"Allowed: {allowed}"
                    ],
                    outcome="Invalid action for current task.",
                ),
            )

        # ---- payload validation -------------------------------------------
        payload_errors = self._validate_payload(action)
        if payload_errors:
            return (
                self._build_observation(),
                -0.1,
                False,
                StepInfo(
                    action_valid=False,
                    validation_errors=payload_errors,
                    outcome="Action rejected due to missing payload fields.",
                ),
            )

        # ---- loop / repeat detection --------------------------------------
        if self._is_repeat_action(action):
            self._state.step_number += 1
            return (
                self._build_observation(),
                -0.05,
                False,
                StepInfo(
                    action_valid=True,
                    outcome="Repeated action – no state change.",
                ),
            )

        # ---- route to task handler ----------------------------------------
        outcome = self._route_action(action)

        if not outcome.get("skip_state_update"):
            self._state = outcome["updated_state"]

        self._state.action_history.append(
            {
                "step": self._state.step_number,
                "action_type": action.action_type.value,
                "payload": action.payload,
                "outcome": outcome.get("description", ""),
                "success": outcome.get("success", False),
            }
        )
        self._state.step_number += 1

        # ---- compute reward -----------------------------------------------
        reward = self._reward_engine.compute(action, outcome, self._state)
        self._state.reward_history.append(reward)
        self._state.cumulative_reward += reward

        # ---- check done ---------------------------------------------------
        done = self._check_done()

        # ---- grade if done ------------------------------------------------
        grader_score: Optional[float] = None
        episode_summary: Optional[dict] = None

        if done:
            grader_score = self._grader_engine.grade(self._state, self._scenario)
            episode_summary = {
                "scenario_id": self._state.scenario_id,
                "difficulty": self._state.difficulty.value,
                "task_type": self._state.task_type.value,
                "grader_score": grader_score,
                "total_steps": self._state.step_number,
                "cumulative_reward": round(self._state.cumulative_reward, 4),
            }
            self._state.status = EpisodeStatus.COMPLETED

        obs = self._build_observation()
        obs.done = done

        info = StepInfo(
            action_valid=True,
            outcome=outcome.get("description", ""),
            grader_score=grader_score,
            episode_summary=episode_summary,
        )
        return obs, reward, done, info

    def state(self) -> InternalState:
        """
        Return the full internal state including hidden fields.

        IMPORTANT: This is intended for testing and debugging only.
        Do NOT pass this to the agent during inference – it contains
        ground-truth labels that must remain hidden.
        """
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() first.")
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_initial_state(
        self,
        scenario: ScenarioDefinition,
        episode_id: str,
    ) -> InternalState:
        ctx = scenario.context
        gt = scenario.ground_truth
        task = scenario.task_type

        state = InternalState(
            episode_id=episode_id,
            scenario_id=scenario.scenario_id,
            task_type=task,
            difficulty=scenario.difficulty,
            max_steps=scenario.max_steps,
        )

        if task == TaskType.REPORT_CLASSIFICATION:
            state.report_state = ReportState(
                report_id=ctx["report_id"],
                correct_label=gt["correct_label"],
            )

        elif task == TaskType.BILLING_VERIFICATION:
            state.billing_state = BillingState(
                claim_id=ctx["claim_id"],
                correct_code=gt["correct_code"],
                code_is_valid=gt["code_is_valid"],
                insurance_covers=gt["insurance_covers"],
                has_discrepancy=gt["has_discrepancy"],
                correct_final_decision=gt["correct_final_decision"],
            )

        elif task == TaskType.BLOOD_BANK:
            inventory = dict(ctx["inventory"])
            state.blood_bank_state = BloodBankState(
                request_id=ctx["request_id"],
                requested_type=ctx["requested_type"],
                requested_units=ctx["requested_units"],
                urgency=ctx["urgency"],
                inventory=inventory,
                initial_inventory=dict(inventory),
                expiry_data=dict(ctx.get("expiry_status", {})),
                should_discard_types=gt.get("should_discard_types", []),
                should_request_restock=gt.get("should_request_restock", False),
                correct_allocation_type=gt["correct_allocation_type"],
            )

        elif task == TaskType.ICU_BED_SCHEDULING:
            state.icu_state = IcuSchedulingState(
                ward_id=ctx["ward_id"],
                total_beds=ctx["total_beds"],
                available_beds=ctx["available_beds"],
                available_bed_ids=list(ctx.get("available_bed_ids", [])),
                pending_requests=list(ctx["pending_requests"]),
                current_occupancy=list(ctx["current_occupancy"]),
                correct_patient_to_admit=gt["correct_patient_to_admit"],
                correct_bed_id=gt["correct_bed_id"],
                should_discharge_patient=gt.get("should_discharge_patient"),
                should_escalate=gt.get("should_escalate", False),
                requires_ventilator_bed=gt.get("requires_ventilator_bed", False),
                requires_isolation_bed=gt.get("requires_isolation_bed", False),
            )

        return state

    def _build_observation(self) -> Observation:
        """
        Construct a sanitised Observation from the internal state.
        Ground-truth / hidden fields are explicitly excluded.
        """
        s = self._state
        ctx = self._scenario.context
        allowed = TASK_ALLOWED_ACTIONS[s.task_type]

        if s.task_type == TaskType.REPORT_CLASSIFICATION:
            rs = s.report_state
            task_context = {
                "report_id": ctx["report_id"],
                "report_text": ctx["report_text"],
                "available_labels": ctx["available_labels"],
                "classified": rs.classified,
            }

        elif s.task_type == TaskType.BILLING_VERIFICATION:
            bs = s.billing_state
            task_context = {
                "claim_id": ctx["claim_id"],
                "billing_code": ctx["billing_code"],
                "reported_procedure": ctx["reported_procedure"],
                "insurance_plan_id": ctx["insurance_plan_id"],
                "claimed_amount": ctx["claimed_amount"],
                "report_category": ctx["report_category"],
                "code_validated": bs.code_validated,
                "insurance_verified": bs.insurance_verified,
                "discrepancy_flagged": bs.discrepancy_flagged,
                "final_decision": bs.final_decision,
            }

        elif s.task_type == TaskType.BLOOD_BANK:
            bb = s.blood_bank_state
            task_context = {
                "request_id": ctx["request_id"],
                "requested_type": ctx["requested_type"],
                "requested_units": ctx["requested_units"],
                "urgency": ctx["urgency"],
                "inventory": dict(bb.inventory),
                "expiry_status": dict(bb.expiry_data),
                "restock_requested": bb.restock_requested,
                "allocation_fulfilled": bb.allocation_fulfilled,
            }

        elif s.task_type == TaskType.ICU_BED_SCHEDULING:
            icu = s.icu_state
            task_context = {
                "ward_id": icu.ward_id,
                "total_beds": icu.total_beds,
                "available_beds": icu.available_beds,
                "available_bed_ids": list(icu.available_bed_ids),
                "pending_requests": list(icu.pending_requests),
                "current_occupancy": list(icu.current_occupancy),
                # Progress flags (visible, not ground truth)
                "patient_assessed": icu.patient_assessed,
                "assessed_patient_id": icu.assessed_patient_id,
                "bed_assigned": icu.bed_assigned,
                "assigned_bed_id": icu.assigned_bed_id,
                "patient_discharged": icu.patient_discharged,
                "issue_escalated": icu.issue_escalated,
                "admission_confirmed": icu.admission_confirmed,
            }
        else:
            task_context = {}

        return Observation(
            task_type=s.task_type,
            step_number=s.step_number,
            max_steps=s.max_steps,
            episode_id=s.episode_id,
            available_actions=allowed,
            task_context=task_context,
            done=(s.status != EpisodeStatus.RUNNING),
        )

    def _validate_payload(self, action: Action) -> list[str]:
        """Return list of validation error strings; empty means OK."""
        required = REQUIRED_PAYLOAD_KEYS.get(action.action_type.value, [])
        return [
            f"Missing required payload key: '{k}'"
            for k in required
            if k not in action.payload
        ]

    def _is_repeat_action(self, action: Action) -> bool:
        """True if this action is identical to the most recent action."""
        if not self._state.action_history:
            return False
        last = self._state.action_history[-1]
        return (
            last["action_type"] == action.action_type.value
            and last["payload"] == action.payload
        )

    def _route_action(self, action: Action) -> dict:
        """Dispatch action to the appropriate task handler."""
        task = self._state.task_type
        if task == TaskType.REPORT_CLASSIFICATION:
            return ReportClassificationTask.handle(action, self._state)
        if task == TaskType.BILLING_VERIFICATION:
            return BillingVerificationTask.handle(action, self._state)
        if task == TaskType.BLOOD_BANK:
            return BloodBankTask.handle(action, self._state)
        if task == TaskType.ICU_BED_SCHEDULING:
            return IcuSchedulingTask.handle(action, self._state)
        raise ValueError(f"No handler for task type: {task}")

    def _check_done(self) -> bool:
        """Return True when the episode should end."""
        s = self._state

        if s.step_number >= s.max_steps:
            return True

        if s.task_type == TaskType.REPORT_CLASSIFICATION:
            return s.report_state.classified

        if s.task_type == TaskType.BILLING_VERIFICATION:
            return s.billing_state.final_decision is not None

        if s.task_type == TaskType.BLOOD_BANK:
            bb = s.blood_bank_state
            if not bb.allocation_fulfilled:
                return False
            if bb.should_request_restock and not bb.restock_requested:
                return False
            return True

        if s.task_type == TaskType.ICU_BED_SCHEDULING:
            return s.icu_state.admission_confirmed

        return False
