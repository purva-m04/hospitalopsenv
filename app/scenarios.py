"""
app/scenarios.py
================
ScenarioLoader – reads scenario JSON files from disk and validates them
against the ScenarioDefinition Pydantic model.

All scenarios live under  scenarios/<scenario_id>.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from app.models import ScenarioDefinition


# Scenarios that ship with the project
KNOWN_SCENARIOS = [
    "report_easy",
    "report_medium",
    "report_hard",
    "billing_easy",
    "billing_medium",
    "billing_hard",
    "bloodbank_easy",
    "bloodbank_medium",
    "bloodbank_hard",
    "icu_easy",
    "icu_medium",
    "icu_hard",
]


class ScenarioLoader:
    """
    Loads and caches scenario definitions from the scenarios/ directory.

    Usage
    -----
    loader = ScenarioLoader("scenarios")
    scenario = loader.load("billing_medium")
    """

    def __init__(self, scenarios_dir: str = "scenarios") -> None:
        self._dir = Path(scenarios_dir)
        self._cache: Dict[str, ScenarioDefinition] = {}

        if not self._dir.exists():
            raise FileNotFoundError(
                f"Scenarios directory not found: {self._dir.resolve()}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, scenario_id: str) -> ScenarioDefinition:
        """
        Load and return the ScenarioDefinition for *scenario_id*.
        Results are cached so repeated loads don't re-read disk.

        Raises
        ------
        FileNotFoundError : if the JSON file doesn't exist
        ValueError        : if the JSON doesn't match the schema
        """
        if scenario_id in self._cache:
            return self._cache[scenario_id]

        path = self._dir / f"{scenario_id}.json"
        if not path.exists():
            available = self.list_available()
            raise FileNotFoundError(
                f"Scenario '{scenario_id}' not found at {path}. "
                f"Available scenarios: {available}"
            )

        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        try:
            scenario = ScenarioDefinition(**raw)
        except Exception as exc:
            raise ValueError(
                f"Scenario '{scenario_id}' failed schema validation: {exc}"
            ) from exc

        # Sanity: the filename should match the id declared inside the JSON
        if scenario.scenario_id != scenario_id:
            raise ValueError(
                f"scenario_id mismatch: file is '{scenario_id}' "
                f"but JSON declares '{scenario.scenario_id}'"
            )

        self._cache[scenario_id] = scenario
        return scenario

    def list_available(self) -> list[str]:
        """Return sorted list of scenario IDs (JSON files present on disk)."""
        return sorted(
            p.stem for p in self._dir.glob("*.json")
        )

    def preload_all(self) -> None:
        """Load every known scenario into cache (useful at startup)."""
        for sid in self.list_available():
            try:
                self.load(sid)
            except Exception as exc:
                print(f"[ScenarioLoader] WARNING: could not load '{sid}': {exc}")
