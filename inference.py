"""
inference.py  --  HospitalOpsEnv root inference script
"""
from __future__ import annotations
import json, os, traceback

API_BASE_URL  = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN      = os.environ.get("HF_TOKEN")
USE_HEURISTIC = os.environ.get("USE_HEURISTIC", "0") == "1"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ALL_SCENARIOS = [
    "report_easy",    "report_medium",    "report_hard",
    "billing_easy",   "billing_medium",   "billing_hard",
    "bloodbank_easy", "bloodbank_medium", "bloodbank_hard",
    "icu_easy",       "icu_medium",       "icu_hard",
]

SCORE_MIN = 0.001
SCORE_MAX = 0.999

def clamp_score(score) -> float:
    if score is None:
        return SCORE_MIN
    v = float(score)
    # Must be STRICTLY between 0 and 1 — never exactly 0.0 or 1.0
    if v <= 0.0:
        return SCORE_MIN
    if v >= 1.0:
        return SCORE_MAX
    return round(max(SCORE_MIN, min(SCORE_MAX, v)), 4)

SYSTEM_PROMPT = """\
You are an intelligent hospital operations agent navigating a simulation environment.
At each step you receive a JSON observation describing a hospital task.
Respond ONLY with a JSON object with fields action_type and payload.
"""

# Per-episode mutable state for heuristic (reset each episode)
_episode_state: dict = {}

def heuristic_action(obs: dict) -> dict:
    task = obs["task_type"]
    ctx  = obs["task_context"]

    # ---- Report Classification ------------------------------------------
    if task == "report_classification":
        from app.utils import heuristic_classify
        label = heuristic_classify(ctx.get("report_text", ""))
        return {"action_type": "classify_report",
                "payload": {"report_id": ctx["report_id"], "label": label}}

    # ---- Billing Verification ------------------------------------------
    if task == "billing_verification":
        claim_id = ctx["claim_id"]
        if not ctx.get("code_validated"):
            return {"action_type": "validate_billing_code",
                    "payload": {"claim_id": claim_id, "code": ctx["billing_code"]}}
        if not ctx.get("insurance_verified"):
            return {"action_type": "verify_insurance",
                    "payload": {"claim_id": claim_id, "plan_id": ctx["insurance_plan_id"]}}
        billing_code = ctx["billing_code"]
        plan_id      = ctx["insurance_plan_id"]
        report_cat   = ctx.get("report_category", "")
        code_looks_invalid      = not billing_code[:2].isdigit()
        plan_is_restrictive     = plan_id in ("PLAN-NONE", "PLAN-BASIC")
        plan_wont_cover_imaging = plan_is_restrictive and report_cat == "imaging"
        if code_looks_invalid:
            if not ctx.get("discrepancy_flagged"):
                return {"action_type": "flag_discrepancy",
                        "payload": {"claim_id": claim_id,
                                    "reason": "Billing code appears invalid or not in schedule"}}
            return {"action_type": "reject_claim",
                    "payload": {"claim_id": claim_id, "reason": "Invalid billing code detected"}}
        if plan_wont_cover_imaging:
            return {"action_type": "reject_claim",
                    "payload": {"claim_id": claim_id,
                                "reason": "Insurance plan does not cover imaging procedures"}}
        return {"action_type": "approve_claim", "payload": {"claim_id": claim_id}}

    # ---- Blood Bank Management -----------------------------------------
    if task == "blood_bank":
        request_id = ctx["request_id"]
        req_type   = ctx["requested_type"]
        req_units  = ctx["requested_units"]
        inventory  = ctx["inventory"]
        expiry     = ctx.get("expiry_status", {})

        # FIX: Track which blood types we've already discarded this episode
        # using the episode_id so we don't re-discard after inventory drops.
        # The env keeps expiry_status static in obs even after discarding,
        # so we must track discarded types ourselves to avoid the repeat-action
        # detector (or inventory misread) causing an infinite discard loop.
        ep_id = obs.get("episode_id", "")
        discarded_set = _episode_state.get(ep_id, {}).get("discarded_types", set())

        for btype, expiring_count in expiry.items():
            if btype in discarded_set:
                # Already sent a discard for this type this episode — skip
                continue
            live = inventory.get(btype, 0)
            if expiring_count > 0 and live >= expiring_count:
                # Record that we're discarding this type now
                if ep_id not in _episode_state:
                    _episode_state[ep_id] = {}
                _episode_state[ep_id].setdefault("discarded_types", set()).add(btype)
                return {"action_type": "discard_expired",
                        "payload": {"blood_type": btype,
                                    "units_to_discard": expiring_count}}

        # Step 2: allocate exact type if available
        available = inventory.get(req_type, 0)
        if available >= req_units:
            return {"action_type": "allocate_blood",
                    "payload": {"request_id": request_id,
                                "blood_type": req_type, "units": req_units}}

        # Step 3: O- universal donor
        o_neg = inventory.get("O-", 0)
        if o_neg >= req_units and req_type != "O-":
            return {"action_type": "use_compatible_type",
                    "payload": {"request_id": request_id,
                                "substitute_type": "O-", "units": req_units}}

        # Step 4: partial allocation
        if available > 0 and not ctx.get("allocation_fulfilled"):
            return {"action_type": "allocate_blood",
                    "payload": {"request_id": request_id,
                                "blood_type": req_type, "units": available}}

        # Step 5: restock
        if not ctx.get("restock_requested"):
            return {"action_type": "request_restock",
                    "payload": {"blood_type": req_type, "units_requested": 10}}

        # Step 6: partial O-
        if o_neg > 0 and req_type == "O-":
            return {"action_type": "allocate_blood",
                    "payload": {"request_id": request_id,
                                "blood_type": req_type, "units": o_neg}}

        if o_neg > 0 and req_type != "O-":
            return {"action_type": "use_compatible_type",
                    "payload": {"request_id": request_id,
                                "substitute_type": "O-", "units": o_neg}}

        return {"action_type": "allocate_blood",
                "payload": {"request_id": request_id, "blood_type": req_type, "units": 1}}

    # ---- ICU Bed Scheduling --------------------------------------------
    if task == "icu_bed_scheduling":
        pending        = ctx.get("pending_requests", [])
        available_beds = ctx.get("available_bed_ids", [])
        occupancy      = ctx.get("current_occupancy", [])
        assessed       = ctx.get("assessed_patient_id")
        assigned       = ctx.get("assigned_bed_id")
        discharged     = ctx.get("patient_discharged", False)
        escalated      = ctx.get("issue_escalated", False)

        # Find step-down ready patient
        stepdown_patient = None
        stepdown_bed     = None
        for bed in occupancy:
            if bed.get("ready_for_stepdown"):
                stepdown_patient = bed["patient_id"]
                stepdown_bed     = bed["bed_id"]
                break

        # Highest priority pending patient
        top_patient = None
        if pending:
            top_patient = sorted(pending, key=lambda x: -x.get("priority_score", 0))[0]

        needs_ventilator = top_patient.get("ventilator_required", False) if top_patient else False
        needs_isolation  = top_patient.get("isolation_required",  False) if top_patient else False

        # Step 1: assess
        if not assessed and top_patient:
            return {"action_type": "assess_patient",
                    "payload": {"patient_id": top_patient["patient_id"]}}

        # Step 2: discharge step-down patient to free a bed
        if assessed and not available_beds and stepdown_patient and not discharged:
            return {"action_type": "discharge_patient",
                    "payload": {"patient_id": stepdown_patient}}

        # FIX: escalate_issue only requires {"reason": "..."} in its payload.
        # The env's REQUIRED_PAYLOAD_KEYS = {"escalate_issue": ["reason"]}.
        # After discharge the beds list may still be empty in this obs tick;
        # only escalate if special infra is needed AND not yet escalated.
        if assessed and (needs_ventilator or needs_isolation) and not escalated:
            parts = []
            if needs_ventilator: parts.append("ventilator")
            if needs_isolation:  parts.append("isolation")
            reason = "Patient requires special infrastructure: " + ", ".join(parts)
            return {"action_type": "escalate_issue",
                    "payload": {"reason": reason}}

        # Step 4: assign bed
        if assessed and not assigned:
            beds_to_use = available_beds or ([stepdown_bed] if stepdown_bed else [])
            if beds_to_use:
                return {"action_type": "assign_bed",
                        "payload": {"patient_id": assessed, "bed_id": beds_to_use[0]}}

        # Step 5: confirm admission
        if assessed and assigned:
            return {"action_type": "confirm_admission",
                    "payload": {"patient_id": assessed, "bed_id": assigned}}

        if top_patient:
            return {"action_type": "assess_patient",
                    "payload": {"patient_id": top_patient["patient_id"]}}

    return {"action_type": "classify_report",
            "payload": {"report_id": "UNKNOWN", "label": "billing"}}


def call_llm(client, conversation):
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=conversation,
        temperature=0.0, max_tokens=300, timeout=20)
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        lines = [l for l in content.split("\n") if not l.strip().startswith("```")]
        content = "\n".join(lines).strip()
    return json.loads(content)

def llm_action(client, conversation, obs):
    user_msg = {"role": "user", "content": (
        "Current observation:\n" + json.dumps(obs, indent=2) +
        "\n\nRespond with your next action as JSON.")}
    conversation = conversation + [user_msg]
    action_dict  = call_llm(client, conversation)
    conversation = conversation + [{"role": "assistant", "content": json.dumps(action_dict)}]
    return action_dict, conversation

def run_episode(env, client, scenario_id, use_heuristic):
    total_reward = 0.0
    steps_taken  = 0
    grader_score = None
    step_rewards = []
    MAX_STEPS    = 15
    try:
        obs        = env.reset(scenario_id)
        episode_id = obs.episode_id
        # Clean up old episode state to avoid memory growth
        keys_to_drop = [k for k in _episode_state if k != episode_id]
        for k in keys_to_drop:
            _episode_state.pop(k, None)
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
            except Exception:
                action_dict = heuristic_action(obs_dict)
            try:
                from app.models import Action, ActionType
                action = Action(
                    action_type=ActionType(action_dict["action_type"]),
                    payload=action_dict.get("payload", {}),
                    episode_id=episode_id)
            except Exception as exc:
                print(f"[STEP] step={steps_taken+1} action=error reward=0.00 done=false error={str(exc)[:80]}", flush=True)
                break
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps_taken  += 1
            step_rewards.append(reward)
            error_str = info.outcome if not info.action_valid else "null"
            print(f"[STEP] step={steps_taken} action={action.action_type.value} reward={reward:.2f} done={'true' if done else 'false'} error={error_str}", flush=True)
            if done:
                grader_score = clamp_score(info.grader_score)
                break
    except Exception:
        traceback.print_exc()
    finally:
        success     = clamp_score(grader_score) >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in step_rewards)
        print(f"[END] success={'true' if success else 'false'} steps={steps_taken} rewards={rewards_str}", flush=True)
    return {"scenario_id": scenario_id, "grader_score": clamp_score(grader_score),
            "total_reward": round(total_reward, 4), "steps": steps_taken}

def main():
    from app.env import HospitalOpsEnv
    client        = None
    use_heuristic = USE_HEURISTIC
    if not use_heuristic:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            print(f"[INFO] LLM agent: {MODEL_NAME} @ {API_BASE_URL}", flush=True)
        except ImportError:
            print("[WARNING] openai not installed, using heuristic.", flush=True)
            use_heuristic = True
    else:
        print("[INFO] USE_HEURISTIC=1 - running deterministic heuristic agent.", flush=True)
    env     = HospitalOpsEnv(scenarios_dir="scenarios")
    results = []
    for scenario_id in ALL_SCENARIOS:
        try:
            results.append(run_episode(env, client, scenario_id, use_heuristic))
        except Exception:
            traceback.print_exc()
            results.append({"scenario_id": scenario_id, "grader_score": SCORE_MIN,
                            "total_reward": 0.0, "steps": 0})
    for r in results:
        r["grader_score"] = clamp_score(r.get("grader_score"))
    avg = clamp_score(sum(r["grader_score"] for r in results) / len(results)) if results else SCORE_MIN
    output = {"results": results, "average_grader_score": round(avg, 4)}
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DONE] Average grader score: {avg:.4f}", flush=True)

if __name__ == "__main__":
    main()
