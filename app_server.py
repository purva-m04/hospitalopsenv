"""
app_server.py
=============
Thin FastAPI wrapper exposing HospitalOpsEnv over HTTP.
Used as the entrypoint for Hugging Face Spaces (Docker SDK).

Endpoints
---------
GET  /health           → liveness check
GET  /scenarios        → list all available scenario IDs
POST /reset            → start a new episode
POST /step             → take one action in the active episode
GET  /state            → full internal state (debug only)
POST /run_inference    → run heuristic agent across all scenarios
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi import Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.env import HospitalOpsEnv
from app.models import Action, ActionType
from app.scenarios import ScenarioLoader

app = FastAPI(
    title="HospitalOpsEnv",
    description="OpenEnv-compatible hospital operations simulation.",
    version="1.0.0",
)

# Single shared environment instance (stateful per server process)
_env = HospitalOpsEnv(scenarios_dir="scenarios")
_loader = ScenarioLoader(scenarios_dir="scenarios")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    scenario_id: str = "report_easy"  # default fallback for empty body {}


class StepRequest(BaseModel):
    action_type: str
    payload: Dict[str, Any] = {}
    episode_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Liveness probe for Hugging Face Spaces."""
    return {"status": "ok", "environment": "HospitalOpsEnv", "version": "1.0.0"}


@app.get("/scenarios")
def list_scenarios():
    """Return all available scenario IDs."""
    return {"scenarios": _loader.list_available()}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    """
    Start a new episode with the given scenario_id.
    Returns the initial observation.
    """
    if req is None:
        req = ResetRequest()
    try:
        obs = _env.reset(req.scenario_id)
        return obs.model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step")
def step(req: StepRequest):
    """
    Apply one action to the current episode.
    Returns observation, reward, done, and info.
    """
    try:
        action = Action(
            action_type=ActionType(req.action_type),
            payload=req.payload,
            episode_id=req.episode_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")

    try:
        obs, reward, done, info = _env.step(action)
        return {
            "observation": obs.model_dump(mode="json"),
            "reward": reward,
            "done": done,
            "info": info.model_dump(mode="json"),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
def get_state():
    """
    Return full internal state (including hidden fields).
    For debugging / testing only — do not expose to agents.
    """
    try:
        state = _env.state()
        return state.model_dump(mode="json")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/run_inference")
def run_inference(use_heuristic: bool = False):
    """
    Run the inference script across all scenarios.
    Returns stdout output and the results JSON.
    """
    import json as _json
    import os

    env_copy = os.environ.copy()
    if use_heuristic:
        env_copy["USE_HEURISTIC"] = "0"

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            env=env_copy,
            timeout=300,
        )
        output = result.stdout + result.stderr

        # Try to read results.json if created
        scores = {}
        try:
            with open("results.json", "r") as fh:
                scores = _json.load(fh)
        except FileNotFoundError:
            pass

        return {"output": output, "scores": scores}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference run timed out.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))