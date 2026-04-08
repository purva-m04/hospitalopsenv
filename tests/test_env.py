"""
tests/test_env.py
=================
Integration tests for HospitalOpsEnv: reset / step / state API.
Run with: pytest tests/test_env.py -v
"""

import pytest
from app.env import HospitalOpsEnv
from app.models import Action, ActionType, EpisodeStatus, TaskType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env():
    return HospitalOpsEnv(scenarios_dir="scenarios")


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------


def test_reset_returns_observation(env):
    obs = env.reset("report_easy")
    assert obs.task_type == TaskType.REPORT_CLASSIFICATION
    assert obs.step_number == 0
    assert obs.done is False
    assert "report_text" in obs.task_context


def test_reset_clears_previous_episode(env):
    obs1 = env.reset("report_easy")
    obs2 = env.reset("billing_easy")
    assert obs1.episode_id != obs2.episode_id
    assert obs2.task_type == TaskType.BILLING_VERIFICATION


def test_reset_unknown_scenario_raises(env):
    with pytest.raises(FileNotFoundError):
        env.reset("nonexistent_scenario")


def test_reset_hidden_fields_not_in_observation(env):
    obs = env.reset("report_easy")
    ctx = obs.task_context
    assert "correct_label" not in ctx


# ---------------------------------------------------------------------------
# step() – guard tests
# ---------------------------------------------------------------------------


def test_step_before_reset_raises(env):
    action = Action(action_type=ActionType.CLASSIFY_REPORT,
                    payload={"report_id": "RPT-001", "label": "lab"},
                    episode_id="fake")
    with pytest.raises(RuntimeError):
        env.step(action)


def test_step_wrong_episode_id_rejected(env):
    obs = env.reset("report_easy")
    action = Action(
        action_type=ActionType.CLASSIFY_REPORT,
        payload={"report_id": "RPT-001", "label": "lab"},
        episode_id="wrong-id",
    )
    _, reward, done, info = env.step(action)
    assert info.action_valid is False
    assert reward == -0.1
    assert done is False


def test_step_wrong_task_action_rejected(env):
    obs = env.reset("report_easy")
    action = Action(
        action_type=ActionType.APPROVE_CLAIM,
        payload={"claim_id": "CLM-101"},
        episode_id=obs.episode_id,
    )
    _, reward, done, info = env.step(action)
    assert info.action_valid is False
    assert reward == -0.1


def test_step_missing_payload_key_rejected(env):
    obs = env.reset("report_easy")
    action = Action(
        action_type=ActionType.CLASSIFY_REPORT,
        payload={"report_id": "RPT-001"},   # missing 'label'
        episode_id=obs.episode_id,
    )
    _, reward, done, info = env.step(action)
    assert info.action_valid is False
    assert any("label" in e for e in info.validation_errors)


def test_step_repeat_action_penalised(env):
    obs = env.reset("billing_easy")
    action = Action(
        action_type=ActionType.VALIDATE_BILLING_CODE,
        payload={"claim_id": "CLM-101", "code": "85025"},
        episode_id=obs.episode_id,
    )
    env.step(action)  # first time – normal
    _, reward, _, info = env.step(action)  # repeat
    assert reward == -0.05
    assert "Repeated" in info.outcome


# ---------------------------------------------------------------------------
# Full episode tests
# ---------------------------------------------------------------------------


def test_report_easy_correct_classification(env):
    obs = env.reset("report_easy")
    action = Action(
        action_type=ActionType.CLASSIFY_REPORT,
        payload={"report_id": "RPT-001", "label": "lab"},
        episode_id=obs.episode_id,
    )
    obs2, reward, done, info = env.step(action)
    assert done is True
    assert reward == 1.0
    assert info.grader_score == 1.0
    assert obs2.done is True


def test_report_easy_wrong_classification(env):
    obs = env.reset("report_easy")
    action = Action(
        action_type=ActionType.CLASSIFY_REPORT,
        payload={"report_id": "RPT-001", "label": "imaging"},
        episode_id=obs.episode_id,
    )
    _, reward, done, info = env.step(action)
    assert done is True
    assert reward == -0.5
    assert info.grader_score is not None
    assert info.grader_score < 1.0


def test_billing_easy_full_correct_episode(env):
    obs = env.reset("billing_easy")
    eid = obs.episode_id

    actions = [
        Action(action_type=ActionType.VALIDATE_BILLING_CODE,
               payload={"claim_id": "CLM-101", "code": "85025"},
               episode_id=eid),
        Action(action_type=ActionType.VERIFY_INSURANCE,
               payload={"claim_id": "CLM-101", "plan_id": "PLAN-BLUE"},
               episode_id=eid),
        Action(action_type=ActionType.APPROVE_CLAIM,
               payload={"claim_id": "CLM-101"},
               episode_id=eid),
    ]
    done = False
    final_info = None
    for a in actions:
        obs, reward, done, info = env.step(a)
        if done:
            final_info = info
            break

    assert done is True
    assert final_info.grader_score == 1.0


def test_billing_medium_reject_correct(env):
    obs = env.reset("billing_medium")
    eid = obs.episode_id

    env.step(Action(action_type=ActionType.VALIDATE_BILLING_CODE,
                    payload={"claim_id": "CLM-202", "code": "71046"},
                    episode_id=eid))
    env.step(Action(action_type=ActionType.VERIFY_INSURANCE,
                    payload={"claim_id": "CLM-202", "plan_id": "PLAN-BASIC"},
                    episode_id=eid))
    _, _, done, info = env.step(Action(
        action_type=ActionType.REJECT_CLAIM,
        payload={"claim_id": "CLM-202", "reason": "Insurance does not cover imaging"},
        episode_id=eid,
    ))
    assert done is True
    assert info.grader_score == 1.0


def test_billing_hard_full_correct_episode(env):
    obs = env.reset("billing_hard")
    eid = obs.episode_id

    env.step(Action(action_type=ActionType.VALIDATE_BILLING_CODE,
                    payload={"claim_id": "CLM-303", "code": "99214"},
                    episode_id=eid))
    env.step(Action(action_type=ActionType.VERIFY_INSURANCE,
                    payload={"claim_id": "CLM-303", "plan_id": "PLAN-BLUE"},
                    episode_id=eid))
    env.step(Action(action_type=ActionType.FLAG_DISCREPANCY,
                    payload={"claim_id": "CLM-303",
                             "reason": "Invalid code and category mismatch"},
                    episode_id=eid))
    _, _, done, info = env.step(Action(
        action_type=ActionType.REJECT_CLAIM,
        payload={"claim_id": "CLM-303", "reason": "Invalid billing code"},
        episode_id=eid,
    ))
    assert done is True
    assert info.grader_score == 1.0


def test_bloodbank_easy_single_allocation(env):
    obs = env.reset("bloodbank_easy")
    eid = obs.episode_id
    _, _, done, info = env.step(Action(
        action_type=ActionType.ALLOCATE_BLOOD,
        payload={"request_id": "BLD-001", "blood_type": "A+", "units": 2},
        episode_id=eid,
    ))
    assert done is True
    assert info.grader_score == 1.0


def test_bloodbank_over_allocation_blocked(env):
    obs = env.reset("bloodbank_easy")
    _, reward, done, info = env.step(Action(
        action_type=ActionType.ALLOCATE_BLOOD,
        payload={"request_id": "BLD-001", "blood_type": "A+", "units": 999},
        episode_id=obs.episode_id,
    ))
    assert info.action_valid is True   # structurally valid, but…
    assert reward == -0.3              # penalised for impossible allocation
    assert done is False


def test_step_after_done_raises(env):
    obs = env.reset("report_easy")
    action = Action(
        action_type=ActionType.CLASSIFY_REPORT,
        payload={"report_id": "RPT-001", "label": "lab"},
        episode_id=obs.episode_id,
    )
    env.step(action)  # episode ends
    with pytest.raises(RuntimeError):
        env.step(action)


# ---------------------------------------------------------------------------
# state() tests
# ---------------------------------------------------------------------------


def test_state_exposes_hidden_fields(env):
    obs = env.reset("report_easy")
    s = env.state()
    assert s.report_state is not None
    assert s.report_state.correct_label == "lab"  # hidden from obs


def test_state_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.state()


def test_episode_summary_populated_on_done(env):
    obs = env.reset("report_easy")
    _, _, done, info = env.step(Action(
        action_type=ActionType.CLASSIFY_REPORT,
        payload={"report_id": "RPT-001", "label": "lab"},
        episode_id=obs.episode_id,
    ))
    assert done is True
    assert info.episode_summary is not None
    assert "grader_score" in info.episode_summary
    assert "total_steps" in info.episode_summary
