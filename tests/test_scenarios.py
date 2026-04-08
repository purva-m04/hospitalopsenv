"""
tests/test_scenarios.py
=======================
Validates that every scenario JSON file loads correctly and conforms
to the ScenarioDefinition schema.

Run with: pytest tests/test_scenarios.py -v
"""

import pytest
from app.scenarios import ScenarioLoader, KNOWN_SCENARIOS
from app.models import ScenarioDefinition, TaskType

loader = ScenarioLoader("scenarios")


@pytest.mark.parametrize("scenario_id", KNOWN_SCENARIOS)
def test_scenario_loads_without_error(scenario_id):
    scenario = loader.load(scenario_id)
    assert isinstance(scenario, ScenarioDefinition)


@pytest.mark.parametrize("scenario_id", KNOWN_SCENARIOS)
def test_scenario_id_matches_filename(scenario_id):
    scenario = loader.load(scenario_id)
    assert scenario.scenario_id == scenario_id


@pytest.mark.parametrize("scenario_id", KNOWN_SCENARIOS)
def test_scenario_has_required_fields(scenario_id):
    s = loader.load(scenario_id)
    assert s.max_steps > 0
    assert s.description
    assert s.context
    assert s.ground_truth


@pytest.mark.parametrize("scenario_id", [
    "report_easy", "report_medium", "report_hard"
])
def test_report_scenarios_have_correct_label(scenario_id):
    s = loader.load(scenario_id)
    assert "correct_label" in s.ground_truth
    assert s.ground_truth["correct_label"] in [
        "lab", "imaging", "prescription", "billing", "discharge"
    ]
    assert "report_text" in s.context
    assert "available_labels" in s.context


@pytest.mark.parametrize("scenario_id", [
    "billing_easy", "billing_medium", "billing_hard"
])
def test_billing_scenarios_have_required_ground_truth(scenario_id):
    s = loader.load(scenario_id)
    gt = s.ground_truth
    for key in ["correct_code", "code_is_valid", "insurance_covers",
                "has_discrepancy", "correct_final_decision"]:
        assert key in gt, f"Missing ground_truth key '{key}' in {scenario_id}"
    assert gt["correct_final_decision"] in ["approve_claim", "reject_claim"]


@pytest.mark.parametrize("scenario_id", [
    "bloodbank_easy", "bloodbank_medium", "bloodbank_hard"
])
def test_bloodbank_scenarios_have_required_ground_truth(scenario_id):
    s = loader.load(scenario_id)
    gt = s.ground_truth
    for key in ["correct_allocation_type", "should_discard_types",
                "should_request_restock"]:
        assert key in gt, f"Missing ground_truth key '{key}' in {scenario_id}"
    assert s.context["requested_type"] in [
        "A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"
    ]


@pytest.mark.parametrize("scenario_id", [
    "report_easy", "billing_easy", "bloodbank_easy"
])
def test_easy_scenarios_have_lower_step_limit(scenario_id):
    easy = loader.load(scenario_id)
    hard_id = scenario_id.replace("easy", "hard")
    hard = loader.load(hard_id)
    assert easy.max_steps <= hard.max_steps


def test_unknown_scenario_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        loader.load("does_not_exist")


def test_list_available_returns_all_known():
    available = loader.list_available()
    for sid in KNOWN_SCENARIOS:
        assert sid in available


def test_loader_caches_scenarios():
    s1 = loader.load("report_easy")
    s2 = loader.load("report_easy")
    assert s1 is s2  # same object from cache
