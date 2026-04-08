"""
app/tasks.py
============
Task handler classes – one per hospital task.

Each handler exposes a single static method:
    handle(action, state) -> dict

The returned dict always contains:
    updated_state : InternalState  (deep-copied, mutated copy)
    success       : bool
    description   : str            (human-readable outcome)

Extra keys may be present per task (e.g. "correct" for report classification).
"""

from __future__ import annotations

from copy import deepcopy

from app.models import (
    Action,
    ActionType,
    InternalState,
)
from app.utils import (
    CPT_CODE_TABLE,
    INSURANCE_COVERAGE_TABLE,
    BLOOD_COMPATIBILITY,
    VALID_REPORT_LABELS,
    is_compatible_blood,
)


# ---------------------------------------------------------------------------
# Task 1 – Medical Report Classification
# ---------------------------------------------------------------------------


class ReportClassificationTask:
    """
    The agent receives a report text and must call classify_report with the
    correct label.  Only one action is allowed; the episode ends immediately.
    """

    @staticmethod
    def handle(action: Action, state: InternalState) -> dict:
        updated = deepcopy(state)
        rs = updated.report_state

        label: str = action.payload.get("label", "")

        if label not in VALID_REPORT_LABELS:
            return {
                "updated_state": updated,
                "success": False,
                "correct": False,
                "description": (
                    f"Invalid label '{label}'. "
                    f"Must be one of {VALID_REPORT_LABELS}."
                ),
            }

        rs.submitted_label = label
        rs.classified = True
        correct = label == rs.correct_label

        return {
            "updated_state": updated,
            "success": True,
            "correct": correct,
            "description": (
                f"Report classified as '{label}'. "
                + ("✓ Correct." if correct else f"✗ Incorrect (expected '{rs.correct_label}').")
            ),
        }


# ---------------------------------------------------------------------------
# Task 2 – Billing & Insurance Verification
# ---------------------------------------------------------------------------


class BillingVerificationTask:
    """
    Multi-step task. The agent must:
      1. validate_billing_code
      2. verify_insurance
      3. flag_discrepancy  (only when a discrepancy exists)
      4. approve_claim OR reject_claim

    Attempting to approve without first validating the code is penalised.
    """

    @staticmethod
    def handle(action: Action, state: InternalState) -> dict:
        at = action.action_type
        updated = deepcopy(state)
        bs = updated.billing_state
        p = action.payload

        # ---- validate_billing_code ----------------------------------------
        if at == ActionType.VALIDATE_BILLING_CODE:
            code: str = p.get("code", "")
            entry = CPT_CODE_TABLE.get(code)
            is_valid: bool = entry is not None and entry[2] is True

            bs.code_validated = True
            bs.code_correct_result = is_valid

            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Billing code '{code}' evaluated: "
                    + ("valid ✓" if is_valid else "invalid ✗")
                    + (f" ({entry[0]})" if entry else " (unknown code)")
                ),
            }

        # ---- verify_insurance -----------------------------------------------
        if at == ActionType.VERIFY_INSURANCE:
            plan_id: str = p.get("plan_id", "")
            covered_codes = INSURANCE_COVERAGE_TABLE.get(plan_id, [])
            covers: bool = bs.correct_code in covered_codes

            bs.insurance_verified = True
            bs.insurance_correct_result = covers

            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Plan '{plan_id}' "
                    + ("covers ✓" if covers else "does NOT cover ✗")
                    + " this procedure."
                ),
            }

        # ---- flag_discrepancy -----------------------------------------------
        if at == ActionType.FLAG_DISCREPANCY:
            reason: str = p.get("reason", "")
            if not reason.strip():
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "flag_discrepancy requires a non-empty 'reason'.",
                }
            bs.discrepancy_flagged = True
            return {
                "updated_state": updated,
                "success": True,
                "description": f"Discrepancy flagged: {reason}",
            }

        # ---- approve_claim ---------------------------------------------------
        if at == ActionType.APPROVE_CLAIM:
            if not bs.code_validated:
                return {
                    "updated_state": updated,
                    "success": False,
                    "skip_state_update": True,
                    "description": (
                        "Cannot approve claim: billing code not yet validated. "
                        "Call validate_billing_code first."
                    ),
                }
            bs.final_decision = "approve_claim"
            return {
                "updated_state": updated,
                "success": True,
                "description": "Claim approved.",
            }

        # ---- reject_claim ---------------------------------------------------
        if at == ActionType.REJECT_CLAIM:
            reason = p.get("reason", "no reason given")
            bs.final_decision = "reject_claim"
            return {
                "updated_state": updated,
                "success": True,
                "description": f"Claim rejected: {reason}",
            }

        # Fallback
        return {
            "updated_state": updated,
            "success": False,
            "description": f"Unknown billing action: {at}",
        }


# ---------------------------------------------------------------------------
# Task 3 – Blood Bank Management
# ---------------------------------------------------------------------------


class BloodBankTask:
    """
    Inventory management task. The agent must:
      - Discard expired units when required
      - Allocate the correct blood type / units
      - Request restock when stock will fall critically low
      - Use a compatible substitute only when exact type is unavailable
    """

    @staticmethod
    def handle(action: Action, state: InternalState) -> dict:
        at = action.action_type
        updated = deepcopy(state)
        bb = updated.blood_bank_state
        p = action.payload

        # ---- allocate_blood -------------------------------------------------
        if at == ActionType.ALLOCATE_BLOOD:
            btype: str = p.get("blood_type", "")
            try:
                units = int(p.get("units", 0))
            except (TypeError, ValueError):
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "'units' must be an integer.",
                }

            available = bb.inventory.get(btype, 0)
            if units <= 0:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": f"units must be > 0, got {units}.",
                }
            if units > available:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"Cannot allocate {units} unit(s) of {btype}: "
                        f"only {available} available."
                    ),
                }

            bb.inventory[btype] = available - units
            bb.allocated_type = btype
            bb.allocated_units = units
            if units >= bb.requested_units:
                bb.allocation_fulfilled = True

            return {
                "updated_state": updated,
                "success": True,
                "description": f"Allocated {units} unit(s) of {btype}.",
            }

        # ---- use_compatible_type -------------------------------------------
        if at == ActionType.USE_COMPATIBLE_TYPE:
            sub_type: str = p.get("substitute_type", "")
            try:
                units = int(p.get("units", 0))
            except (TypeError, ValueError):
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "'units' must be an integer.",
                }

            if not is_compatible_blood(bb.requested_type, sub_type):
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"⚠ UNSAFE: {sub_type} is NOT compatible "
                        f"for {bb.requested_type} patients."
                    ),
                }

            available = bb.inventory.get(sub_type, 0)
            if units > available:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"Only {available} unit(s) of {sub_type} available, "
                        f"but {units} requested."
                    ),
                }

            bb.inventory[sub_type] = available - units
            bb.allocated_type = sub_type
            bb.allocated_units = units
            if units >= bb.requested_units:
                bb.allocation_fulfilled = True

            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Allocated {units} unit(s) of compatible type "
                    f"{sub_type} (requested: {bb.requested_type})."
                ),
            }

        # ---- request_restock -----------------------------------------------
        if at == ActionType.REQUEST_RESTOCK:
            btype = p.get("blood_type", "")
            try:
                units_req = int(p.get("units_requested", 0))
            except (TypeError, ValueError):
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "'units_requested' must be an integer.",
                }
            if units_req <= 0 or units_req > 50:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "units_requested must be between 1 and 50.",
                }

            bb.restock_requested = True
            # If a partial allocation was already made, restock closes the episode
            if bb.allocated_units > 0 and not bb.allocation_fulfilled:
                bb.allocation_fulfilled = True
            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Restock requested: {units_req} unit(s) of {btype}. "
                    f"Partial allocation documented — episode closed."
                ),
            }

        # ---- discard_expired -----------------------------------------------
        if at == ActionType.DISCARD_EXPIRED:
            btype = p.get("blood_type", "")
            try:
                discard_units = int(p.get("units_to_discard", 0))
            except (TypeError, ValueError):
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "'units_to_discard' must be an integer.",
                }

            available = bb.inventory.get(btype, 0)
            if discard_units <= 0:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "units_to_discard must be > 0.",
                }
            if discard_units > available:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"Cannot discard {discard_units} unit(s) of {btype}: "
                        f"only {available} in inventory."
                    ),
                }

            bb.inventory[btype] = available - discard_units
            bb.units_discarded += discard_units
            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Discarded {discard_units} expired unit(s) of {btype}. "
                    f"Remaining: {bb.inventory[btype]}."
                ),
            }

        # Fallback
        return {
            "updated_state": updated,
            "success": False,
            "description": f"Unknown blood bank action: {at}",
        }


# ---------------------------------------------------------------------------
# Task 4 – ICU Bed Scheduling
# ---------------------------------------------------------------------------


class IcuSchedulingTask:
    """
    Multi-step ICU bed management task. The agent must:
      1. assess_patient   – evaluate the highest-priority pending patient
      2. discharge_patient (if needed) – free a step-down-ready bed
      3. escalate_issue   (if needed) – flag infrastructure gaps (isolation, etc.)
      4. assign_bed       – assign the correct bed to the prioritised patient
      5. confirm_admission – finalise and close the episode

    The episode terminates once confirm_admission is successfully called.
    """

    @staticmethod
    def handle(action: Action, state: InternalState) -> dict:
        at = action.action_type
        updated = deepcopy(state)
        icu = updated.icu_state
        p = action.payload

        # ---- assess_patient ------------------------------------------------
        if at == ActionType.ASSESS_PATIENT:
            patient_id: str = p.get("patient_id", "")
            pending_ids = [r["patient_id"] for r in icu.pending_requests]
            if patient_id not in pending_ids:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"Patient '{patient_id}' not found in pending requests. "
                        f"Available: {pending_ids}"
                    ),
                }
            icu.patient_assessed = True
            icu.assessed_patient_id = patient_id
            req = next(r for r in icu.pending_requests if r["patient_id"] == patient_id)
            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Patient {patient_id} assessed: priority={req['priority_score']}, "
                    f"diagnosis='{req['diagnosis']}', "
                    f"ventilator_required={req['ventilator_required']}, "
                    f"isolation_required={req['isolation_required']}."
                ),
            }

        # ---- discharge_patient ---------------------------------------------
        if at == ActionType.DISCHARGE_PATIENT:
            patient_id = p.get("patient_id", "")
            occupant_ids = [o["patient_id"] for o in icu.current_occupancy]
            if patient_id not in occupant_ids:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"Patient '{patient_id}' is not a current ICU occupant. "
                        f"Current patients: {occupant_ids}"
                    ),
                }
            occupant = next(o for o in icu.current_occupancy if o["patient_id"] == patient_id)
            freed_bed = occupant["bed_id"]
            icu.current_occupancy = [
                o for o in icu.current_occupancy if o["patient_id"] != patient_id
            ]
            icu.available_bed_ids.append(freed_bed)
            icu.available_beds += 1
            icu.patient_discharged = True
            icu.discharged_patient_id = patient_id
            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Patient {patient_id} discharged from bed {freed_bed}. "
                    f"Bed {freed_bed} is now available."
                ),
            }

        # ---- escalate_issue ------------------------------------------------
        if at == ActionType.ESCALATE_ISSUE:
            reason: str = p.get("reason", "")
            if not reason.strip():
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": "escalate_issue requires a non-empty 'reason'.",
                }
            icu.issue_escalated = True
            icu.escalation_reason = reason
            return {
                "updated_state": updated,
                "success": True,
                "description": f"Issue escalated to charge nurse / bed manager: {reason}",
            }

        # ---- assign_bed ----------------------------------------------------
        if at == ActionType.ASSIGN_BED:
            if not icu.patient_assessed:
                return {
                    "updated_state": updated,
                    "success": False,
                    "skip_state_update": True,
                    "description": (
                        "Cannot assign bed: no patient has been assessed. "
                        "Call assess_patient first."
                    ),
                }
            patient_id = p.get("patient_id", "")
            bed_id: str = p.get("bed_id", "")

            pending_ids = [r["patient_id"] for r in icu.pending_requests]
            if patient_id not in pending_ids:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": f"Patient '{patient_id}' not in pending requests.",
                }
            if bed_id not in icu.available_bed_ids:
                return {
                    "updated_state": updated,
                    "success": False,
                    "description": (
                        f"Bed '{bed_id}' is not available. "
                        f"Available beds: {icu.available_bed_ids}"
                    ),
                }
            icu.bed_assigned = True
            icu.assigned_bed_id = bed_id
            icu.assigned_patient_id = patient_id
            icu.available_bed_ids.remove(bed_id)
            icu.available_beds -= 1
            return {
                "updated_state": updated,
                "success": True,
                "description": f"Bed {bed_id} assigned to patient {patient_id}.",
            }

        # ---- confirm_admission ---------------------------------------------
        if at == ActionType.CONFIRM_ADMISSION:
            if not icu.bed_assigned:
                return {
                    "updated_state": updated,
                    "success": False,
                    "skip_state_update": True,
                    "description": (
                        "Cannot confirm admission: no bed has been assigned. "
                        "Call assign_bed first."
                    ),
                }
            icu.admission_confirmed = True
            return {
                "updated_state": updated,
                "success": True,
                "description": (
                    f"Admission confirmed: patient {icu.assigned_patient_id} "
                    f"admitted to bed {icu.assigned_bed_id} in ward {icu.ward_id}."
                ),
            }

        # Fallback
        return {
            "updated_state": updated,
            "success": False,
            "description": f"Unknown ICU scheduling action: {at}",
        }