"""
FinWise OpenEnv — FastAPI Server
Powers the Hugging Face Space. Exposes /reset, /step, /state endpoints.
Must respond HTTP 200 to automated validation pings.
"""

from __future__ import annotations
import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from finwise_env.env import FinWiseEnv
from finwise_env.models import PortfolioAction
from finwise_env.tasks import ALL_TASK_NAMES

app = FastAPI(
    title="FinWise OpenEnv",
    description=(
        "Real-world OpenEnv environment for training and evaluating LLM agents "
        "on Indian retail portfolio advisory tasks."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → FinWiseEnv
_sessions: Dict[str, FinWiseEnv] = {}


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: Optional[str] = "diversify_sector_easy"
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    asset: Optional[str] = None
    amount_inr: Optional[float] = None
    target_weight: Optional[float] = None
    reasoning: Optional[str] = None


class StateRequest(BaseModel):
    session_id: str


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "FinWise OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": ALL_TASK_NAMES,
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks", "/docs"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(_sessions)}


@app.get("/tasks")
async def list_tasks():
    return {"tasks": ALL_TASK_NAMES}


# ─────────────────────────────────────────────
# /reset — start a new episode
# ─────────────────────────────────────────────

@app.post("/reset")
async def reset(request: ResetRequest = None):
    """
    Start a new episode. Returns initial portfolio observation.
    Creates a new session or resets an existing one.
    """
    if request is None:
        request = ResetRequest()

    task_name = request.task_name or "diversify_sector_easy"
    if task_name not in ALL_TASK_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Valid tasks: {ALL_TASK_NAMES}"
        )

    session_id = request.session_id or str(uuid.uuid4())
    env = FinWiseEnv(task_name=task_name)
    observation = env.reset()
    _sessions[session_id] = env

    return {
        "session_id": session_id,
        "task_name": task_name,
        "observation": observation.model_dump(),
        "done": False,
    }


# ─────────────────────────────────────────────
# /step — agent takes one action
# ─────────────────────────────────────────────

@app.post("/step")
async def step(request: StepRequest):
    """
    Agent submits one portfolio action.
    Returns updated observation, reward, done flag, and score info.
    """
    env = _sessions.get(request.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /reset first."
        )

    action = PortfolioAction(
        action_type=request.action_type,
        asset=request.asset,
        amount_inr=request.amount_inr,
        target_weight=request.target_weight,
        reasoning=request.reasoning,
    )

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Clean up session if done
    if result.done:
        _sessions.pop(request.session_id, None)

    return {
        "session_id": request.session_id,
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "reward_breakdown": result.reward_breakdown.model_dump(),
        "done": result.done,
        "info": result.info,
    }


# ─────────────────────────────────────────────
# /state — get current internal state
# ─────────────────────────────────────────────

@app.post("/state")
async def state(request: StateRequest):
    """Return current internal environment state for the session."""
    env = _sessions.get(request.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{request.session_id}' not found. Call /reset first."
        )
    return {"session_id": request.session_id, "state": env.state()}


# ─────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
