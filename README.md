# HospitalOpsEnv

**A rule-based, deterministic OpenEnv environment simulating real-world hospital operational workflows.**

HospitalOpsEnv trains and evaluates AI agents on four critical hospital tasks: medical report classification, billing & insurance verification, blood bank management, and ICU bed scheduling.

---

## Real-World Motivation

Hospitals generate thousands of operational decisions every day. Errors cost lives and money. HospitalOpsEnv provides a safe, reproducible simulation where agents can learn and be evaluated on these exact workflows.

---

## Tasks

| Task | Difficulty | Scenarios |
|------|-----------|-----------|
| Medical Report Classification | Easy / Medium / Hard | report_easy, report_medium, report_hard |
| Billing & Insurance Verification | Easy / Medium / Hard | billing_easy, billing_medium, billing_hard |
| Blood Bank Management | Easy / Medium / Hard | bloodbank_easy, bloodbank_medium, bloodbank_hard |
| ICU Bed Scheduling | Easy / Medium / Hard | icu_easy, icu_medium, icu_hard |

---

## Action Space

- classify_report — label a medical report
- alidate_billing_code — verify a CPT billing code
- erify_insurance — check insurance plan coverage
- lag_discrepancy — flag a billing mismatch
- pprove_claim / eject_claim — final billing decision
- llocate_blood — allocate blood units to a request
- equest_restock — restock blood inventory
- discard_expired — discard expired blood units
- use_compatible_type — substitute compatible blood type
- ssess_patient — assess ICU patient priority
- ssign_bed — assign an ICU bed
- discharge_patient — discharge a stable patient
- confirm_admission — confirm ICU admission
- escalate_issue — escalate infrastructure issue

---

## Observation Space

Structured JSON containing: 	ask_type, 	ask_context, vailable_actions, step_number, done, episode_id.

---

## Setup

`ash
pip install -r requirements.txt
python inference.py
`

## Docker

`ash
docker build -t hospitalopsenv .
docker run -p 7860:7860 hospitalopsenv
`

## Environment Variables

| Variable | Description |
|----------|-------------|
| OPENAI_API_KEY | Your OpenAI API key |
| API_BASE_URL | LLM API base URL |
| MODEL_NAME | Model identifier |
| HF_TOKEN | Hugging Face token |
| USE_HEURISTIC | Set to 1 to use heuristic agent |

## Baseline Scores

Run python inference.py with USE_HEURISTIC=1 to reproduce baseline scores. Results saved to esults.json.
