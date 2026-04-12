"""
inference.py
============
Root-level inference script for HospitalOpsEnv.
Runs an LLM-driven agent through all 12 scenarios and reports grader scores.
Falls back to a deterministic heuristic agent when no API key is available.

Environment variables
---------------------
API_BASE_URL    : API base URL (default: https://api.openai.com/v1)
MODEL_NAME      : model to use  (default: gpt-4o-mini)
HF_TOKEN        : HuggingFace token (required)
USE_HEURISTIC   : set to "1" to force heuristic agent (no LLM calls)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Optional

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

API_BASE_URL:  str  = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:    str  = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN             = os.environ.get("HF_TOKEN")
USE_HEURISTIC: bool = os.environ.get("USE_HEURISTIC", "0") == "1"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ALL_SCENARIOS = [
    "report_easy",    "report_medium",    "report_hard",
    "billing_easy",   "billing_medium",   "billing_hard",
    "bloodbank_easy", "bloodbank_medium", "bloodbank_hard",
    "icu_easy",       "icu_medium",       "icu_hard",
]

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an intelligent hospital operations agent navigating a simulation environment.

At each step you receive a JSON observation describing a hospital task.
You must respond with a single JSON object representing your next action.

Your response MUST be valid JSON with exactly these two fields:
{
  "action_type": "<one of the available action strings>",
  "payload": { <required keys for that action> }
}

Action types and their required payload keys:
  classify_report         -> {"report_id": str, "label": str}
  validate_billing_code   -> {"claim_id": str, "code": str}
  verify_insurance        -> {"claim_id": str, "plan_id": str}
  flag_discrepancy        -> {"claim_id": str, "reason": str}
  approve_claim           -> {"claim_id": str}
  reject_claim            -> {"claim_id": str, "reason": str}
  allocate_blood          -> {"request_id": str, "blood_type": str, "units": int}
  request_restock         -> {"blood_type": str, "units_requested": int}
  discard_expired         -> {"blood_type": str, "units_to_discard": int}
  use_compatible_type     -> {"request_id": str, "substitute_type": str, "units": int}
  assess_patient          -> {"patient_id": str}
  assign_bed              -> {"patient_id": str, "bed_id": str}
  confirm_admission       -> {"patient_id": str, "bed_id": str}
  discharge_patient       -> {"patient_id": str}
  escalate_issue          -> {"issue_type": str, "details": str}

Valid report labels: lab, imaging, prescription, billing, discharge

Rules:
- Only use actions listed in available_actions in the observation.
- For billing tasks: always validate_billing_code before approve_claim.
- For blood bank: check if the exact blood type is available before substituting.
- If a blood type is unavailable, consider use_compatible_type with a safe substitute.
- Universal donor for red blood cells: O-
- For ICU: assess patient first, then assign bed, then confirm admission.

Respond ONLY with a JSON object. No explanation, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Heuristic fallback agent
# ---------------------------------------------------------------------------

def heuristic_action(obs: dict) -> dict:
    task = obs["task_type"]
    ctx  = obs["task_context"]

    # ---- Report Classification -------------------------------------------
    if task == "report_classification":
        from app.utils import heuristic_classify
        label = heuristic_classify(ctx.get("report_text", ""))
        return {
            "action_type": "classify_report",
            "payload": {"report_id": ctx["report_id"], "label": label},
        }

    # ---- Billing Verification -------------------------------------------
    if task == "billing_verification":
        claim_id = ctx["claim_id"]

        if not ctx.get("code_validated"):
            return {
                "action_type": "validate_billing_code",
                "payload": {"claim_id": claim_id, "code": ctx["billing_code"]},
            }

        if not ctx.get("insurance_verified"):
            return {
                "action_type": "verify_insurance",
                "payload": {"claim_id": claim_id, "plan_id": ctx["insurance_plan_id"]},
            }

        billing_code = ctx["billing_code"]
        plan_id      = ctx["insurance_plan_id"]
        report_cat   = ctx.get("report_category", "")

        code_looks_invalid      = not billing_code[:2].isdigit()
        plan_is_restrictive     = plan_id in ("PLAN-NONE", "PLAN-BASIC")
        plan_wont_cover_imaging = plan_is_restrictive and report_cat == "imaging"

        if code_looks_invalid:
            if not ctx.get("discrepancy_flagged"):
                return {
                    "action_type": "flag_discrepancy",
                    "payload": {"claim_id": claim_id,
                                "reason": "Billing code appears invalid or not in schedule"},
                }
            return {
                "action_type": "reject_claim",
                "payload": {"claim_id": claim_id, "reason": "Invalid billing code detected"},
            }

        if plan_wont_cover_imaging:
            return {
                "action_type": "reject_claim",
                "payload": {"claim_id": claim_id,
                            "reason": "Insurance plan does not cover imaging procedures"},
            }

        return {
            "action_type": "approve_claim",
            "payload": {"claim_id": claim_id},
        }

    # ---- Blood Bank Management ------------------------------------------
    if task == "blood_bank":
        request_id = ctx["request_id"]
        req_type   = ctx["requested_type"]
        req_units  = ctx["requested_units"]
        inventory  = ctx["inventory"]
        expiry     = ctx.get("expiry_status", {})

        for btype, expiring_count in expiry.items():
            live_available = inventory.get(btype, 0)
            if expiring_count > 0 and live_available > 0:
                units_to_discard = min(expiring_count, live_available)
                return {
                    "action_type": "discard_expired",
                    "payload": {"blood_type": btype, "units_to_discard": units_to_discard},
                }

        available = inventory.get(req_type, 0)
        if available >= req_units:
            return {
                "action_type": "allocate_blood",
                "payload": {"request_id": request_id, "blood_type": req_type, "units": req_units},
            }

        o_neg = inventory.get("O-", 0)
        if o_neg >= req_units and req_type != "O-":
            return {
                "action_type": "use_compatible_type",
                "payload": {"request_id": request_id, "substitute_type": "O-", "units": req_units},
            }

        if available > 0:
            return {
                "action_type": "allocate_blood",
                "payload": {"request_id": request_id, "blood_type": req_type, "units": available},
            }

        if not ctx.get("restock_requested"):
            return {
                "action_type": "request_restock",
                "payload": {"blood_type": req_type, "units_requested": 10},
            }

        return {
            "action_type": "allocate_blood",
            "payload": {"request_id": request_id, "blood_type": req_type, "units": 1},
        }

    # ---- ICU Bed Scheduling ---------------------------------------------
    if task == "icu_bed_scheduling":
        pending       = ctx.get("pending_requests", [])
        available_beds = ctx.get("available_bed_ids", [])
        assessed      = ctx.get("assessed_patient_id")
        assigned      = ctx.get("assigned_bed_id")
        should_discharge = ctx.get("should_discharge_patient")
        should_escalate  = ctx.get("should_escalate", False)

        # Step 1: discharge a patient if needed to free a bed
        if should_discharge and not ctx.get("discharge_done"):
            return {
                "action_type": "discharge_patient",
                "payload": {"patient_id": should_discharge},
            }

        # Step 2: escalate if required
        if should_escalate and not ctx.get("escalation_done"):
            return {
                "action_type": "escalate_issue",
                "payload": {"issue_type": "infrastructure", "details": "ICU resource escalation required"},
            }

        # Step 3: assess highest priority patient
        if pending and not assessed:
            patient = sorted(pending, key=lambda x: -x.get("priority_score", 0))[0]
            return {
                "action_type": "assess_patient",
                "payload": {"patient_id": patient["patient_id"]},
            }

        # Step 4: assign bed
        if assessed and not assigned and available_beds:
            return {
                "action_type": "assign_bed",
                "payload": {"patient_id": assessed, "bed_id": available_beds[0]},
            }

        # Step 5: confirm admission
        if assessed and assigned:
            return {
                "action_type": "confirm_admission",
                "payload": {"patient_id": assessed, "bed_id": assigned},
            }

        # Fallback
        if pending:
            return {
                "action_type": "assess_patient",
                "payload": {"patient_id": pending[0]["patient_id"]},
            }

    return {"action_type": "classify_report",
            "payload": {"report_id": "UNKNOWN", "label": "billing"}}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def call_llm(client, conversation: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        temperature=0.0,
        max_tokens=300,
        timeout=20,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        lines = [l for l in content.split("\n") if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    return json.loads(content)


def llm_action(client, conversation: list[dict], obs: dict) -> tuple[dict, list[dict]]:
    user_msg = {
        "role": "user",
        "content": (
            "Current observation:\n"
            + json.dumps(obs, indent=2)
            + "\n\nRespond with your next action as JSON."
        ),
    }
    conversation = conversation + [user_msg]
    action_dict  = call_llm(client, conversation)
    conversation = conversation + [{"role": "assistant", "content": json.dumps(action_dict)}]
    return action_dict, conversation


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, client, scenario_id: str, use_heuristic: bool) -> dict:
    total_reward = 0.0
    steps_taken  = 0
    grader_score = 0.001
    step_rewards = []
    MAX_STEPS    = 10

    try:
        obs        = env.reset(scenario_id)
        episode_id = obs.episode_id

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        print(f"[START] task={obs.task_type.value} env=hospitalopsenv model={MODEL_NAME}",
              flush=True)

        while not obs.done and steps_taken < MAX_STEPS:
            obs_dict = obs.model_dump(mode="json")

            try:
                if use_heuristic or not client:
                    action_dict = heuristic_action(obs_dict)
                else:
                    action_dict, conversation = llm_action(client, conversation, obs_dict)
            except Exception as exc:
                action_dict = heuristic_action(obs_dict)

            try:
                from app.models import Action, ActionType
                action = Action(
                    action_type=ActionType(action_dict["action_type"]),
                    payload=action_dict.get("payload", {}),
                    episode_id=episode_id,
                )
            except Exception as exc:
                print(
                    f"[STEP] step={steps_taken+1} action=error "
                    f"reward=0.00 done=false error={str(exc)[:80]}",
                    flush=True
                )
                break

            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps_taken  += 1
            step_rewards.append(reward)

            error_str = info.outcome if not info.action_valid else "null"
            print(
                f"[STEP] step={steps_taken} "
                f"action={action.action_type.value} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={error_str}",
                flush=True
            )

            if done:
                grader_score = max(0.001, min(0.999, info.grader_score or 0.001))
                break

    except Exception as exc:
        traceback.print_exc()

    finally:
        success     = grader_score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={steps_taken} "
            f"rewards={rewards_str}",
            flush=True
        )

    return {
        "scenario_id":  scenario_id,
        "grader_score": grader_score,
        "total_reward": round(total_reward, 4),
        "steps":        steps_taken,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from app.env import HospitalOpsEnv

    client        = None
    use_heuristic = USE_HEURISTIC

    if not use_heuristic:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            print(f"[INFO] LLM agent: {MODEL_NAME} @ {API_BASE_URL}", flush=True)
        except ImportError:
            print("[WARNING] openai package not installed — using heuristic agent.", flush=True)
            use_heuristic = True
    else:
        print("[INFO] USE_HEURISTIC=1 — running deterministic heuristic agent.", flush=True)

    env     = HospitalOpsEnv(scenarios_dir="scenarios")
    results = []

    for scenario_id in ALL_SCENARIOS:
        try:
            result = run_episode(env, client, scenario_id, use_heuristic)
            results.append(result)
        except Exception as exc:
            traceback.print_exc()
            results.append({
                "scenario_id":  scenario_id,
                "grader_score": 0.001,
                "total_reward": 0.0,
                "steps":        0,
            })

    total_score = 0.0
    for r in results:
        score = r.get("grader_score", 0.001)
        score = max(0.001, min(0.999, score))
        total_score += score

    avg = total_score / len(results) if results else 0.001

    output = {
        "results": results,
        "average_grader_score": round(avg, 4),
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] Average grader score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()