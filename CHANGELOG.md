# HospitalOpsEnv Changelog

## v2.0.0 — April 2026

### 1 · Richer Scenario Clinical Notes (all 9 original scenarios)

All nine scenario JSON files received substantially expanded `context` fields.
Each scenario now contains realistic clinical prose matching the kind of
documents that appear in actual hospital information systems.

| Scenario | What was added |
|---|---|
| `report_easy` | Full CBC-with-diff, CMP, thyroid panel, and fasting lipid panel with reference ranges; lab scientist sign-off |
| `report_medium` | Full CT technique, multi-organ findings, radiologist impression, and a pelvic-cyst follow-up recommendation |
| `report_hard` | Multi-section discharge summary with admission H&P, lab values on admission, imaging reports (2 CXRs), hospital course with IV→oral antibiotic switch, discharge medications, and billing note containing CPT distractor codes |
| `billing_easy` | Expanded `reported_procedure` with ordering-physician context, patient demographics, and in-network lab note |
| `billing_medium` | Expanded `reported_procedure` with ED context; `clinical_notes` field explaining PLAN-BASIC exclusions |
| `billing_hard` | See §2 below |
| `bloodbank_easy` | `clinical_context` with pre-operative crossmatch details, antibody screen result, and transfusion threshold |
| `bloodbank_medium` | `clinical_context` with post-surgical haemorrhage scenario, expiry bag IDs, and SOP reference |
| `bloodbank_hard` | `clinical_context` with MHP trauma activation, unknown blood type rationale, and regional blood-centre escalation note |

---

### 2 · Harder `billing_hard` — Multi-Fault Cross-Reference Challenge

The original `billing_hard` used a trivially-fake code (`FAKE99`) that any
agent could detect by a simple alphanumeric prefix check.

The new scenario uses **CPT 99214** (Office Visit — Moderate Complexity), a
real, syntactically valid code that is looked up successfully by
`validate_billing_code`. Three compounding faults require genuine
cross-field reasoning:

| Fault | Detail |
|---|---|
| **Category mismatch** | Code category is `billing` (office visit); `report_category` is `imaging` (MRI brain) |
| **Amount mismatch** | `claimed_amount` = $1,850 — far above the $150 fee-schedule ceiling for 99214, and also above the $850 ceiling for the actual procedure (CPT 70553) |
| **Coverage nuance** | PLAN-BLUE *does* cover 99214 for legitimate office visits, so `insurance_covers = true` — the agent must still reject because the service-type and amount are wrong |

The heuristic agent in `inference.py` was updated with:
- `_CODE_CATEGORY` mapping (CPT code → service category)
- `_FEE_SCHEDULE_MAX` per-category ceiling table
- `_billing_has_discrepancy()` that checks all three fault types
- Descriptive multi-part `reason` string in the `flag_discrepancy` payload

---

### 3 · Task 4 — ICU Bed Scheduling (new)

A fourth task was added end-to-end across all layers of the codebase.

**Three new scenarios:**

| Scenario | Challenge |
|---|---|
| `icu_easy` | 1 bed available, 1 pending patient — assess → assign → confirm |
| `icu_medium` | ICU full; highest-priority trauma patient needs admission; one stable patient (8 days, ready for step-down) must be discharged first to free the bed |
| `icu_hard` | ICU full; critical ARDS patient needs both a **ventilator** bed and **contact isolation** (MRSA+); agent must discharge the step-down patient, **escalate** the isolation infrastructure gap, then assign the correct ventilator bay |

**New models (`app/models.py`):**
- `TaskType.ICU_BED_SCHEDULING` enum value
- Five new `ActionType` values: `ASSESS_PATIENT`, `ASSIGN_BED`, `DISCHARGE_PATIENT`, `CONFIRM_ADMISSION`, `ESCALATE_ISSUE`
- `IcuSchedulingState` Pydantic model tracking ward state, pending requests, current occupancy, and all agent-progress flags

**New task handler (`app/tasks.py` — `IcuSchedulingTask`):**
- `assess_patient` — validates patient is in pending list; records assessment
- `discharge_patient` — removes occupant from ward, frees the bed ID
- `escalate_issue` — records infrastructure alert with mandatory reason string
- `assign_bed` — requires prior assessment; validates bed is available; assigns patient
- `confirm_admission` — requires prior assignment; terminal action that ends episode

**Grader (`app/graders.py` — `_grade_icu`):**

| Component | Weight |
|---|---|
| Correct patient assessed | 0.15 |
| Discharge handling (correct patient or none needed) | 0.15 |
| Escalation handling (raised when needed, not raised when not) | 0.15 |
| Correct bed assigned | 0.25 |
| Correct patient admitted | 0.20 |
| Admission confirmed | 0.10 |

**Rewards (`app/rewards.py` — `_reward_icu`):**
- `assess_patient`: +0.3 correct patient, −0.1 wrong patient
- `discharge_patient`: +0.3 correct, −0.2 wrong patient or unnecessary
- `escalate_issue`: +0.3 real issue, −0.2 false escalation
- `assign_bed`: +0.5 both correct, +0.2 right patient/wrong bed, −0.3 both wrong
- `confirm_admission`: +1.0 perfect episode, +0.5 right assignment but missed auxiliary steps, +0.1 otherwise

**env.py:**
- `TASK_ALLOWED_ACTIONS` and `REQUIRED_PAYLOAD_KEYS` extended
- `_build_initial_state` handles `ICU_BED_SCHEDULING`
- `_build_observation` exposes sanitised ICU state (no ground-truth fields)
- `_route_action` dispatches to `IcuSchedulingTask`
- `_check_done` terminates on `admission_confirmed = True`

**openenv.yaml:** Updated to v2.0.0; ICU task and all 3 scenarios registered.

---

### 4 · Time-Decay Reward (blood bank & billing)

Correct actions in the two time-sensitive tasks now receive an additional
bonus that decays linearly from **+0.10 at step 0** to **0.0 at step 6**:

```
decay_bonus = 0.10 × max(0, 1 − step / 6)
```

The decay bonus is only added to *correct* actions (positive base rewards).
Negative rewards (wrong decisions, penalties) are unaffected.

This creates a mild urgency gradient without making the task unsolvable: an
agent that takes a few exploratory steps still earns the full base reward,
but an agent that acts correctly immediately earns slightly more.

The helper `_time_decay_bonus(step)` lives at module level in `rewards.py`
and is directly importable for testing.

---

### 5 · `inference.py` Updates

**ICU heuristic agent (`heuristic_action` — new `icu_bed_scheduling` branch):**
1. Select the highest-priority patient by `priority_score`
2. If not yet assessed → `assess_patient`
3. If ICU full and not yet discharged → find the `ready_for_stepdown` patient
   with the most `days_in_icu` → `discharge_patient`
4. If patient requires isolation (`isolation_required: true`) → `escalate_issue`
5. Once a bed is free → `assign_bed` (first available bed)
6. Once bed is assigned → `confirm_admission`

**Harder billing heuristic (`heuristic_action` — `billing_verification` branch):**
- Replaced the single `code_looks_invalid` alphanumeric check with the full
  three-fault `_billing_has_discrepancy()` function
- Multi-part reason string in `flag_discrepancy` payload names each fault
  detected (category mismatch, amount mismatch, structural invalidity)
- `_billing_should_reject()` combines all rejection signals

**Report classification heuristic (`classify_report_text`):**
- Added title-level signal boosters (+5 to the relevant category) for
  `discharge summary`, `radiology report`, and `laboratory report` headers,
  preventing rich multi-section documents from being mis-classified by
  incidental keyword overlap

**`SYSTEM_PROMPT`:** Extended with ICU action types, payload signatures,
and decision rules for prioritisation, discharge selection, isolation
escalation, and ventilator-bed matching.

**`ALL_SCENARIOS`:** Extended to include `icu_easy`, `icu_medium`, `icu_hard`.
