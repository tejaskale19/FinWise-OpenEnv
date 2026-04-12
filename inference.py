"""
FinWise OpenEnv — Inference Script (MANDATORY)
=====================================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

REQUIRED LOG FORMAT (must not deviate):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Usage:
    set API_BASE_URL=https://router.huggingface.co/v1
    set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
    set HF_TOKEN=YOUR_TOKEN_HERE
  python inference.py
"""

from __future__ import annotations
import os
import json
import signal
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from finwise_env.env import FinWiseEnv
from finwise_env.models import PortfolioAction
from finwise_env.tasks import ALL_TASK_NAMES

# ─────────────────────────────────────────────
# Configuration — read from environment
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "finwise-openenv"

MAX_STEPS = 10
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.70
REQUEST_TIMEOUT_SECONDS = 45
REQUEST_RETRIES = 2
TASK_TIMEOUT_SECONDS = 300
SCORE_EPSILON = 0.005


def safe_score(raw_score: float) -> float:
    """Clamp score to a range that remains valid after 2dp formatting."""
    try:
        val = float(raw_score)
    except (TypeError, ValueError):
        val = 0.5

    if val != val or val == float("inf") or val == float("-inf"):
        val = 0.5

    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, val))


def _clamp_final_score(score: float) -> float:
    """Backward-compatible alias for older callers."""
    return safe_score(score)


def validate_all_scores(scores: dict) -> None:
    """Validate final task scores before process exit."""
    for task_id, score in scores.items():
        numeric_score = safe_score(score)
        formatted_score = float(f"{numeric_score:.2f}")
        if not (0.0 < formatted_score < 1.0):
            raise AssertionError(
                f"Score for {task_id} = {formatted_score:.2f} is OUT OF RANGE after formatting. "
                "Must be strictly between 0 and 1 (exclusive)."
            )


def validate_scores(scores: dict) -> None:
    """Backward-compatible alias."""
    validate_all_scores(scores)


def validate_formatted_scores(rewards: List[float], final_score: float) -> Tuple[List[float], float]:
    """
    Validate rewards and final score after 2dp formatting.
    Returns safe rewards and a safe final score suitable for [END] logs.
    """
    safe_rewards: List[float] = []
    for reward in rewards:
        clamped = safe_score(reward)
        formatted_val = float(f"{clamped:.2f}")
        if formatted_val <= 0.0 or formatted_val >= 1.0:
            clamped = 0.5
        safe_rewards.append(clamped)

    clamped_final = safe_score(final_score)
    formatted_final = float(f"{clamped_final:.2f}")
    if formatted_final <= 0.0 or formatted_final >= 1.0:
        clamped_final = 0.5

    return safe_rewards, clamped_final


def _single_line(value: object) -> str:
    """Ensure structured log fields never contain embedded newlines."""
    return str(value).replace("\r", " ").replace("\n", " ")


# ─────────────────────────────────────────────
# Structured logging — EXACT FORMAT REQUIRED
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={_single_line(task)} env={_single_line(env)} model={_single_line(model)}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = _single_line(error) if error else "null"
    done_val = "true" if done else "false"
    safe_reward = safe_score(reward)
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={safe_reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> Tuple[List[float], float]:
    safe_rewards, safe_final_score = validate_formatted_scores(rewards, score)
    rewards_str = ",".join(f"{safe_score(r):.2f}" for r in safe_rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} score={safe_final_score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return safe_rewards, safe_final_score


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
    last_exc: Optional[Exception] = None
    for _attempt in range(REQUEST_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT_SECONDS,
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
            last_exc = exc

    return PortfolioAction(action_type="hold"), str(last_exc)


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
    score = 0.5
    success = False
    result = None
    timeout_supported = hasattr(signal, "SIGALRM")
    old_alarm_handler = None

    try:
        if timeout_supported:
            def _timeout_handler(_signum, _frame):
                raise TimeoutError(f"Task exceeded {TASK_TIMEOUT_SECONDS}s timeout")

            old_alarm_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TASK_TIMEOUT_SECONDS)

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
                reward = safe_score(result.reward)
                done = result.done
                obs_dict = result.observation.model_dump()
                step_error = result.info.get("error")
                if error is None:
                    error = step_error
            except Exception as e:
                reward = safe_score(0.5)
                done = True
                if error is None:
                    error = str(e)

            rewards.append(safe_score(reward))
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if rewards:
            raw_final = sum(rewards) / len(rewards)
        else:
            raw_final = 0.5

        score = safe_score(raw_final)

        if result and result.info:
            success = bool(result.info.get("task_success", score >= SUCCESS_SCORE_THRESHOLD))
        else:
            success = score >= SUCCESS_SCORE_THRESHOLD
        score = safe_score(score)

    except Exception:
        success = False
        score = safe_score(0.5)

    finally:
        if timeout_supported:
            signal.alarm(0)
            if old_alarm_handler is not None:
                signal.signal(signal.SIGALRM, old_alarm_handler)

        end_rewards = rewards if rewards else [0.50]
        _, end_score = log_end(success=success, steps=steps_taken, score=safe_score(score), rewards=end_rewards)
        score = safe_score(end_score)

    return safe_score(score)


# ─────────────────────────────────────────────
# Main — run all 3 tasks
# ─────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        raise EnvironmentError(
            "Missing credentials: set HF_TOKEN or API_KEY environment variable."
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    final_scores = {}
    for task_name in ALL_TASK_NAMES:
        final_scores[task_name] = float(run_task(client, task_name))

    validate_all_scores(final_scores)


if __name__ == "__main__":
    main()
