"""
inference.py
============
Root-level inference script for HospitalOpsEnv.

Runs an LLM-driven agent through all 9 scenarios and reports grader scores.
Falls back to a deterministic heuristic agent when no API key is available.

Environment variables
---------------------
OPENAI_API_KEY  : OpenAI (or compatible) API key
API_BASE_URL    : API base URL (default: https://api.openai.com/v1)
MODEL_NAME      : model to use  (default: gpt-4o-mini)
HF_TOKEN        : HuggingFace token (optional)
USE_HEURISTIC   : set to "1" to force heuristic agent (no LLM calls)

Usage
-----
  python inference.py                          # LLM agent
  USE_HEURISTIC=1 python inference.py          # heuristic agent (no API key needed)
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

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL:   str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:     str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN:       str = os.environ.get("HF_TOKEN", "")
USE_HEURISTIC: bool = os.environ.get("USE_HEURISTIC", "0") == "1"

ALL_SCENARIOS = [
    "report_easy",    "report_medium",    "report_hard",
    "billing_easy",   "billing_medium",   "billing_hard",
    "bloodbank_easy", "bloodbank_medium", "bloodbank_hard",
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

Valid report labels: lab, imaging, prescription, billing, discharge

Rules:
- Only use actions listed in available_actions in the observation.
- For billing tasks: always validate_billing_code before approve_claim.
- For blood bank: check if the exact blood type is available before substituting.
- If a blood type is unavailable, consider use_compatible_type with a safe substitute.
- Universal donor for red blood cells: O-

Respond ONLY with a JSON object. No explanation, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Heuristic fallback agent  (fixed: no infinite loop)
# ---------------------------------------------------------------------------

def heuristic_action(obs: dict) -> dict:
    """
    Rule-based agent that handles all 3 tasks without any LLM calls.
    BUG FIX: blood bank uses live inventory counts to decide discards,
    so it never repeats a discard after inventory is already reduced.
    """
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
                "payload": {"claim_id": claim_id,
                            "plan_id": ctx["insurance_plan_id"]},
            }

        billing_code = ctx["billing_code"]
        plan_id      = ctx["insurance_plan_id"]
        report_cat   = ctx.get("report_category", "")

        code_looks_invalid       = not billing_code[:2].isdigit()
        plan_is_restrictive      = plan_id in ("PLAN-NONE", "PLAN-BASIC")
        plan_wont_cover_imaging  = plan_is_restrictive and report_cat == "imaging"

        if code_looks_invalid:
            if not ctx.get("discrepancy_flagged"):
                return {
                    "action_type": "flag_discrepancy",
                    "payload": {"claim_id": claim_id,
                                "reason": "Billing code appears invalid or not in schedule"},
                }
            return {
                "action_type": "reject_claim",
                "payload": {"claim_id": claim_id,
                            "reason": "Invalid billing code detected"},
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
        inventory  = ctx["inventory"]          # LIVE — updates every step
        expiry     = ctx.get("expiry_status", {})

        # Step 1: discard expiring units.
        # KEY FIX: use min(expiring_count, live_available) so we never
        # try to discard more than what is currently in inventory,
        # and once the inventory hits 0 we skip and move on.
        for btype, expiring_count in expiry.items():
            live_available = inventory.get(btype, 0)
            if expiring_count > 0 and live_available > 0:
                units_to_discard = min(expiring_count, live_available)
                return {
                    "action_type": "discard_expired",
                    "payload": {"blood_type": btype,
                                "units_to_discard": units_to_discard},
                }

        # Step 2: allocate exact type
        available = inventory.get(req_type, 0)
        if available >= req_units:
            return {
                "action_type": "allocate_blood",
                "payload": {"request_id": request_id,
                            "blood_type": req_type,
                            "units": req_units},
            }

        # Step 3: O- universal fallback
        o_neg = inventory.get("O-", 0)
        if o_neg >= req_units and req_type != "O-":
            return {
                "action_type": "use_compatible_type",
                "payload": {"request_id": request_id,
                            "substitute_type": "O-",
                            "units": req_units},
            }

        # Step 4: partial allocation (whatever is left)
        if available > 0:
            return {
                "action_type": "allocate_blood",
                "payload": {"request_id": request_id,
                            "blood_type": req_type,
                            "units": available},
            }

        # Step 5: request restock
        if not ctx.get("restock_requested"):
            return {
                "action_type": "request_restock",
                "payload": {"blood_type": req_type, "units_requested": 10},
            }

        # Step 6: truly stuck — force a 0-unit allocate to hit max_steps naturally
        # (env will reject it and eventually done=True via step limit)
        return {
            "action_type": "allocate_blood",
            "payload": {"request_id": request_id,
                        "blood_type": req_type,
                        "units": 1},
        }

    # Fallback
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
    conversation = conversation + [{"role": "assistant",
                                    "content": json.dumps(action_dict)}]
    return action_dict, conversation


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, client, scenario_id: str, use_heuristic: bool) -> dict:
    from app.models import Action, ActionType

    obs        = env.reset(scenario_id)
    episode_id = obs.episode_id

    conversation   = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward   = 0.0
    steps_taken    = 0
    grader_score   = 0.0

    print(f"\n  {'─'*55}")
    print(f"  Scenario : {scenario_id}")
    print(f"  Task     : {obs.task_type.value}")
    print(f"  Episode  : {episode_id}")
    print(f"  {'─'*55}")

    while not obs.done:
        obs_dict = obs.model_dump(mode="json")

        try:
            if use_heuristic or not client:
                action_dict = heuristic_action(obs_dict)
            else:
                action_dict, conversation = llm_action(client, conversation, obs_dict)
        except Exception as exc:
            print(f"    [AGENT ERROR] {exc} — falling back to heuristic")
            action_dict = heuristic_action(obs_dict)

        try:
            action = Action(
                action_type=ActionType(action_dict["action_type"]),
                payload=action_dict.get("payload", {}),
                episode_id=episode_id,
            )
        except Exception as exc:
            print(f"    [ACTION PARSE ERROR] {exc}")
            break

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps_taken  += 1

        status = "✓" if info.action_valid else "✗"
        print(
            f"    Step {steps_taken:>3} {status}  "
            f"{action.action_type.value:<28}  "
            f"reward={reward:+.2f}  "
            f"{info.outcome[:45]}"
        )

        if done:
            grader_score = max(0.001, min(0.999, info.grader_score or 0.001))
            break

    print(f"  {'─'*55}")
    print(f"  Grader score : {grader_score:.3f}   "
          f"Cumulative reward : {total_reward:+.3f}   "
          f"Steps : {steps_taken}")

    return {
        "scenario_id":   scenario_id,
        "grader_score":  grader_score,
        "total_reward":  round(total_reward, 4),
        "steps":         steps_taken,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from app.env import HospitalOpsEnv

    client        = None
    use_heuristic = USE_HEURISTIC

    if not use_heuristic:
        if not OPENAI_API_KEY:
            print("[WARNING] OPENAI_API_KEY not set — switching to heuristic agent.\n")
            use_heuristic = True
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
                print(f"[INFO] LLM agent: {MODEL_NAME} @ {API_BASE_URL}")
            except ImportError:
                print("[WARNING] openai package not installed — using heuristic agent.")
                use_heuristic = True
    else:
        print("[INFO] USE_HEURISTIC=1 — running deterministic heuristic agent.")

    env     = HospitalOpsEnv(scenarios_dir="scenarios")
    results = []

    print(f"\n{'═'*60}")
    print("  HospitalOpsEnv — Inference Run")
    print(f"{'═'*60}")

    for scenario_id in ALL_SCENARIOS:
        try:
            result = run_episode(env, client, scenario_id, use_heuristic)
            results.append(result)
        except Exception as exc:
            print(f"\n  [EPISODE ERROR] {scenario_id}: {exc}")
            traceback.print_exc()
            results.append({
                "scenario_id":  scenario_id,
                "grader_score": 0.0,
                "total_reward": 0.0,
                "steps":        0,
                "error":        str(exc),
            })

    # Summary
    print(f"\n{'═'*60}")
    print("  FINAL SCORE SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Scenario':<28} {'Score':>7}  {'Reward':>8}  {'Steps':>5}")
    print(f"  {'─'*55}")

    total_score = 0.0
    for r in results:
        score = r.get("grader_score", 0.0)
        total_score += score
        err = " [ERROR]" if "error" in r else ""
        print(
            f"  {r['scenario_id']:<28} {score:>7.3f}  "
            f"{r['total_reward']:>+8.3f}  "
            f"{r['steps']:>5}{err}"
        )

    avg = total_score / len(results) if results else 0.0
    print(f"  {'─'*55}")
    print(f"  {'Average score':<28} {avg:>7.3f}")
    print(f"{'═'*60}\n")

    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "agent":                "heuristic" if use_heuristic else f"llm:{MODEL_NAME}",
            "average_grader_score": round(avg, 4),
            "results":              results,
        }, fh, indent=2)
    print("Results saved → results.json")


if __name__ == "__main__":
    main()
