"""
FinWise OpenEnv — Inference Script (MANDATORY)
=====================================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

REQUIRED LOG FORMAT (must not deviate):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Usage:
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    set HF_TOKEN=YOUR_TOKEN_HERE
  python inference.py
"""

from __future__ import annotations
import os
import json
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from finwise_env.env import FinWiseEnv
from finwise_env.graders import clamp_strict_score
from finwise_env.models import PortfolioAction
from finwise_env.tasks import ALL_TASK_NAMES

# ─────────────────────────────────────────────
# Configuration — read from environment
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "finwise-openenv"

MAX_STEPS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.70


def _clamp_final_score(score: float) -> float:
    return clamp_strict_score(score)


# ─────────────────────────────────────────────
# Structured logging — EXACT FORMAT REQUIRED
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# System prompt for the LLM agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are FinWise AI, an expert Indian retail portfolio advisor.
You help investors rebalance their portfolios by choosing one action per step.

AVAILABLE ACTIONS:
- buy_stock: Buy a specific stock. Params: asset (stock symbol), amount_inr (INR amount)
- sell_stock: Sell a specific stock. Params: asset (stock symbol), amount_inr (INR amount)
- increase_sip: Increase monthly SIP. Params: amount_inr (monthly increase in INR)
- decrease_sip: Decrease monthly SIP. Params: amount_inr (monthly decrease in INR)
- buy_mutual_fund: Invest in mutual fund. Params: amount_inr (INR amount)
- sell_mutual_fund: Redeem mutual fund. Params: amount_inr (INR amount)
- rebalance_sector: Rebalance a sector to target weight. Params: asset (sector name), target_weight (0.0-1.0)
- hold: Do nothing this step.

VALID STOCKS: TCS, INFY, WIPRO, HDFCBANK, ICICIBANK, HINDUNILVR, SUNPHARMA, RELIANCE
VALID SECTORS: IT, Banking, FMCG, Pharma, Energy

You MUST respond with ONLY a valid JSON object. No preamble, no explanation outside JSON.
The JSON must have these fields:
{
  "action_type": "string",
  "asset": "string or null",
  "amount_inr": number_or_null,
  "target_weight": number_or_null,
  "reasoning": "one sentence explanation"
}
""").strip()


def build_user_prompt(obs_dict: dict, step: int, task_name: str) -> str:
    """Build the per-step prompt from current observation."""
    sector = obs_dict.get("sector_exposure", {})
    stocks = obs_dict.get("stocks", {})
    return textwrap.dedent(f"""
    TASK: {task_name}
    STEP: {step}

    CURRENT PORTFOLIO STATE:
    - Cash: ₹{obs_dict.get('cash_inr', 0):,.0f}
    - Total Value: ₹{obs_dict.get('total_portfolio_value_inr', 0):,.0f}
    - Monthly SIP: ₹{obs_dict.get('sip_monthly_inr', 0):,.0f}
    - Mutual Fund Value: ₹{obs_dict.get('mutual_fund_value_inr', 0):,.0f}
    - Risk Profile: {obs_dict.get('risk_profile', 'unknown')}
    - Risk Score: {obs_dict.get('risk_score', 0):.2f} (0=low, 1=high)
    - NIFTY Trend: {obs_dict.get('nifty_trend', 'unknown')}
    - Max Drawdown: {obs_dict.get('max_drawdown', 0):.1%}

    SECTOR EXPOSURE:
    - IT: {sector.get('IT', 0):.1%}
    - Banking: {sector.get('Banking', 0):.1%}
    - FMCG: {sector.get('FMCG', 0):.1%}
    - Pharma: {sector.get('Pharma', 0):.1%}
    - Energy: {sector.get('Energy', 0):.1%}

    STOCK HOLDINGS (₹):
    - TCS: {stocks.get('TCS', 0):,.0f}
    - INFY: {stocks.get('INFY', 0):,.0f}
    - WIPRO: {stocks.get('WIPRO', 0):,.0f}
    - HDFCBANK: {stocks.get('HDFCBANK', 0):,.0f}
    - ICICIBANK: {stocks.get('ICICIBANK', 0):,.0f}
    - HINDUNILVR: {stocks.get('HINDUNILVR', 0):,.0f}
    - SUNPHARMA: {stocks.get('SUNPHARMA', 0):,.0f}
    - RELIANCE: {stocks.get('RELIANCE', 0):,.0f}

    GOAL TRACKING:
    - Target Corpus: ₹{obs_dict.get('target_corpus_inr', 0):,.0f}
    - Projected Corpus: ₹{obs_dict.get('projected_corpus_inr', 0):,.0f}
    - Goal Progress: {obs_dict.get('goal_progress', 0):.1%}
    - Investment Horizon: {obs_dict.get('investment_horizon_years', 0)} years

    Last action result: {obs_dict.get('last_action_result', 'N/A')}

    Choose the BEST single action to make progress on the task goal.
    Respond ONLY with a JSON object.
    """).strip()


# ─────────────────────────────────────────────
# LLM agent: get action from model
# ─────────────────────────────────────────────

def get_agent_action(
    client: OpenAI,
    obs_dict: dict,
    step: int,
    task_name: str
) -> Tuple[PortfolioAction, Optional[str]]:
    """Query the LLM and parse response into a PortfolioAction."""
    user_prompt = build_user_prompt(obs_dict, step, task_name)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return PortfolioAction(
            action_type=data.get("action_type", "hold"),
            asset=data.get("asset"),
            amount_inr=data.get("amount_inr"),
            target_weight=data.get("target_weight"),
            reasoning=data.get("reasoning"),
        ), None
    except Exception as exc:
        return PortfolioAction(action_type="hold"), str(exc)


def format_action(action: PortfolioAction) -> str:
    """Create compact action text for strict [STEP] logs."""
    parts = [action.action_type]
    if action.asset is not None:
        parts.append(f"asset={action.asset}")
    if action.amount_inr is not None:
        parts.append(f"amount_inr={action.amount_inr}")
    if action.target_weight is not None:
        parts.append(f"target_weight={action.target_weight}")
    return "|".join(parts)


# ─────────────────────────────────────────────
# Run one task episode
# ─────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run one complete episode for a given task.
    Emits [START], [STEP]×N, [END] to stdout.
    Returns final episode score.
    """
    env = FinWiseEnv(task_name=task_name)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.001
    success = False
    result = None

    try:
        obs = env.reset()
        max_steps = env.state().get("max_steps", MAX_STEPS)
        obs_dict = obs.model_dump()
        done = False

        for step in range(1, max_steps + 1):
            if done:
                break

            action, action_error = get_agent_action(client, obs_dict, step, task_name)
            action_str = format_action(action)
            error = action_error

            try:
                result = env.step(action)
                reward = result.reward
                done = result.done
                obs_dict = result.observation.model_dump()
                step_error = result.info.get("error")
                if error is None:
                    error = step_error
            except Exception as e:
                reward = -0.1
                done = True
                if error is None:
                    error = str(e)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Compute final score from terminal grader
        if result and result.info:
            score = result.info.get("terminal_score", 0.001)
            score = _clamp_final_score(score)
            success = bool(result.info.get("task_success", score >= SUCCESS_SCORE_THRESHOLD))
        else:
            score = 0.001
            success = False
        score = _clamp_final_score(score)

    except Exception:
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return _clamp_final_score(score)


# ─────────────────────────────────────────────
# Main — run all 3 tasks
# ─────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN environment variable not set."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_name in ALL_TASK_NAMES:
        run_task(client, task_name)


if __name__ == "__main__":
    main()
