"""
FinWise OpenEnv — Main Environment
Implements the full OpenEnv interface: reset() / step() / state()
Real-world task: Indian retail portfolio advisory
"""

from __future__ import annotations
import copy
import math
from typing import Any, Dict, Optional, Tuple

from models import (
    PortfolioObservation,
    PortfolioAction,
    RewardBreakdown,
    StepResult,
    SectorExposure,
    StockHoldings,
)
from tasks import TASK_REGISTRY, TaskDefinition
from graders import compute_step_reward, grade_task, clamp_strict_score


# Stock → Sector mapping
STOCK_SECTOR_MAP = {
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HDFCBANK": "Banking",
    "ICICIBANK": "Banking",
    "HINDUNILVR": "FMCG",
    "SUNPHARMA": "Pharma",
    "RELIANCE": "Energy",
}

VALID_STOCKS = list(STOCK_SECTOR_MAP.keys())
VALID_SECTORS = ["IT", "Banking", "FMCG", "Pharma", "Energy"]

VALID_ACTIONS = [
    "buy_stock", "sell_stock",
    "increase_sip", "decrease_sip",
    "buy_mutual_fund", "sell_mutual_fund",
    "rebalance_sector", "hold"
]


class FinWiseEnv:
    """
    FinWise Portfolio Advisory Environment.

    An LLM agent acts as an AI wealth advisor for Indian retail investors.
    At each step it observes the investor's full portfolio state and chooses
    one advisory action. The environment updates the portfolio and returns
    a reward signal reflecting progress toward the task goal.

    Implements the OpenEnv interface:
      reset()       → PortfolioObservation
      step(action)  → StepResult
      state()       → Dict (full internal state)
    """

    ENV_NAME = "finwise-openenv"
    VERSION = "1.0.0"

    def __init__(self, task_name: str = "diversify_sector_easy"):
        if task_name not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Valid tasks: {list(TASK_REGISTRY.keys())}"
            )
        self.task_name = task_name
        self._task: TaskDefinition = TASK_REGISTRY[task_name]
        self._portfolio: Dict = {}
        self._step_count: int = 0
        self._done: bool = False
        self._prev_portfolio: Dict = {}
        self._episode_rewards: list = []

    # ─────────────────────────────────────────────
    # reset() — OpenEnv required method
    # ─────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> PortfolioObservation:
        """
        Start a fresh episode with the task's initial portfolio.
        Returns the initial observation.
        """
        self._portfolio = copy.deepcopy(self._task.initial_portfolio)
        self._step_count = 0
        self._done = False
        self._episode_rewards = []
        self._prev_portfolio = copy.deepcopy(self._portfolio)
        return self._build_observation()

    # ─────────────────────────────────────────────
    # step() — OpenEnv required method
    # ─────────────────────────────────────────────

    def step(self, action: PortfolioAction) -> StepResult:
        """
        Agent takes one portfolio advisory action.
        Returns updated observation, reward, done flag, and info.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._prev_portfolio = copy.deepcopy(self._portfolio)
        error_msg: Optional[str] = None

        # Apply action to portfolio
        try:
            self._apply_action(action)
        except ValueError as e:
            error_msg = str(e)

        # Recompute derived metrics
        self._recompute_metrics()

        self._step_count += 1

        # Compute step reward
        scalar_reward, breakdown_dict = compute_step_reward(
            prev_state=self._prev_portfolio,
            curr_state=self._portfolio,
            action_type=action.action_type,
            task_name=self.task_name,
        )
        self._episode_rewards.append(scalar_reward)

        # Check done: max steps reached or task success
        terminal_score, score_explanation = grade_task(self.task_name, self._portfolio)
        terminal_score = clamp_strict_score(terminal_score)
        task_success = terminal_score >= self._task.success_threshold
        max_steps_reached = self._step_count >= self._task.max_steps
        self._done = task_success or max_steps_reached

        # Build reward breakdown model
        reward_breakdown = RewardBreakdown(
            total_reward=breakdown_dict["total_reward"],
            diversification_delta=breakdown_dict["diversification_delta"],
            corpus_growth_delta=breakdown_dict["corpus_growth_delta"],
            risk_reduction_delta=breakdown_dict["risk_reduction_delta"],
            liquidity_safety_delta=breakdown_dict["liquidity_safety_delta"],
            overconcentration_penalty=breakdown_dict["overconcentration_penalty"],
            churn_penalty=breakdown_dict["churn_penalty"],
            cash_depletion_penalty=breakdown_dict["cash_depletion_penalty"],
            explanation=(
                f"Action={action.action_type}. "
                + (f"Error: {error_msg}. " if error_msg else "")
                + score_explanation
            ),
        )

        obs = self._build_observation(
            last_action_result=error_msg or f"Action '{action.action_type}' applied successfully."
        )

        info = {
            "terminal_score": terminal_score,
            "task_success": task_success,
            "max_steps_reached": max_steps_reached,
            "score_explanation": score_explanation,
            "episode_rewards": self._episode_rewards[:],
            "error": error_msg,
        }

        return StepResult(
            observation=obs,
            reward=scalar_reward,
            reward_breakdown=reward_breakdown,
            done=self._done,
            info=info,
        )

    # ─────────────────────────────────────────────
    # state() — OpenEnv required method
    # ─────────────────────────────────────────────

    def state(self) -> Dict:
        """Return full internal environment state as a dict."""
        return {
            "task_name": self.task_name,
            "step_count": self._step_count,
            "done": self._done,
            "portfolio": copy.deepcopy(self._portfolio),
            "episode_rewards": self._episode_rewards[:],
            "max_steps": self._task.max_steps,
            "success_threshold": self._task.success_threshold,
        }

    # ─────────────────────────────────────────────
    # Internal: apply action to portfolio
    # ─────────────────────────────────────────────

    def _apply_action(self, action: PortfolioAction) -> None:
        """Mutate self._portfolio based on the chosen action."""
        p = self._portfolio

        if action.action_type == "hold":
            return

        elif action.action_type == "buy_stock":
            stock = (action.asset or "").upper()
            amount = action.amount_inr or 0
            if stock not in VALID_STOCKS:
                raise ValueError(f"Unknown stock '{stock}'. Valid: {VALID_STOCKS}")
            if amount <= 0:
                raise ValueError("amount_inr must be positive for buy_stock")
            if p["cash_inr"] < amount:
                amount = p["cash_inr"]  # buy as much as possible
            p["cash_inr"] -= amount
            p["stocks"][stock] = p["stocks"].get(stock, 0) + amount

        elif action.action_type == "sell_stock":
            stock = (action.asset or "").upper()
            amount = action.amount_inr or 0
            if stock not in VALID_STOCKS:
                raise ValueError(f"Unknown stock '{stock}'. Valid: {VALID_STOCKS}")
            current_value = p["stocks"].get(stock, 0)
            if current_value <= 0:
                raise ValueError(f"No holdings in {stock}")
            amount = min(amount, current_value)
            p["stocks"][stock] = current_value - amount
            p["cash_inr"] += amount * 0.99  # 1% transaction cost

        elif action.action_type == "increase_sip":
            amount = action.amount_inr or 0
            if amount <= 0:
                raise ValueError("amount_inr must be positive for increase_sip")
            p["sip_monthly_inr"] += amount

        elif action.action_type == "decrease_sip":
            amount = action.amount_inr or 0
            new_sip = p["sip_monthly_inr"] - amount
            p["sip_monthly_inr"] = max(500, new_sip)  # floor ₹500

        elif action.action_type == "buy_mutual_fund":
            amount = action.amount_inr or 0
            if amount <= 0:
                raise ValueError("amount_inr must be positive for buy_mutual_fund")
            if p["cash_inr"] < amount:
                amount = p["cash_inr"]
            p["cash_inr"] -= amount
            p["mutual_fund_value_inr"] = p.get("mutual_fund_value_inr", 0) + amount

        elif action.action_type == "sell_mutual_fund":
            amount = action.amount_inr or 0
            mf_value = p.get("mutual_fund_value_inr", 0)
            if mf_value <= 0:
                raise ValueError("No mutual fund holdings to sell")
            amount = min(amount, mf_value)
            p["mutual_fund_value_inr"] = mf_value - amount
            p["cash_inr"] += amount * 0.99

        elif action.action_type == "rebalance_sector":
            sector = (action.asset or "").title()
            target_weight = action.target_weight
            if sector not in VALID_SECTORS:
                raise ValueError(f"Unknown sector '{sector}'. Valid: {VALID_SECTORS}")
            if target_weight is None or not (0.0 <= target_weight <= 1.0):
                raise ValueError("target_weight must be between 0.0 and 1.0")
            # Simulate rebalance: sell stocks in over-target sectors, buy in others
            self._simulate_sector_rebalance(sector, target_weight)

        else:
            raise ValueError(f"Unknown action_type '{action.action_type}'. Valid: {VALID_ACTIONS}")

    def _simulate_sector_rebalance(self, target_sector: str, target_weight: float) -> None:
        """Simulate selling stocks in a sector to hit a target weight."""
        p = self._portfolio
        total = self._compute_total_value()
        sector_stocks = [s for s, sec in STOCK_SECTOR_MAP.items() if sec == target_sector]
        current_sector_value = sum(p["stocks"].get(s, 0) for s in sector_stocks)
        target_value = total * target_weight
        delta = current_sector_value - target_value

        if delta > 0:
            # Need to sell from sector
            remaining = delta
            for stock in sector_stocks:
                if remaining <= 0:
                    break
                sell_amount = min(p["stocks"].get(stock, 0), remaining)
                p["stocks"][stock] = p["stocks"].get(stock, 0) - sell_amount
                p["cash_inr"] += sell_amount * 0.99
                remaining -= sell_amount
        elif delta < 0:
            # Need to buy into sector (use available cash)
            buy_total = min(abs(delta), p["cash_inr"])
            per_stock = buy_total / max(len(sector_stocks), 1)
            for stock in sector_stocks:
                p["stocks"][stock] = p["stocks"].get(stock, 0) + per_stock
                p["cash_inr"] -= per_stock

    # ─────────────────────────────────────────────
    # Internal: recompute derived metrics
    # ─────────────────────────────────────────────

    def _compute_total_value(self) -> float:
        p = self._portfolio
        stock_total = sum(p["stocks"].values())
        return p["cash_inr"] + stock_total + p.get("mutual_fund_value_inr", 0)

    def _recompute_metrics(self) -> None:
        p = self._portfolio
        total = self._compute_total_value()
        p["total_portfolio_value_inr"] = total

        # Recompute sector exposure
        sector_values: Dict[str, float] = {s: 0.0 for s in VALID_SECTORS}
        for stock, value in p["stocks"].items():
            sector = STOCK_SECTOR_MAP.get(stock, "Other")
            if sector in sector_values:
                sector_values[sector] += value
        investable = total - p["cash_inr"]
        if investable > 0:
            p["sector_exposure"] = {s: v / investable for s, v in sector_values.items()}
        else:
            p["sector_exposure"] = {s: 0.0 for s in VALID_SECTORS}

        # Recompute risk score: weighted by sector concentration and drawdown
        it = p["sector_exposure"].get("IT", 0)
        banking = p["sector_exposure"].get("Banking", 0)
        concentration_risk = (it + banking) / 2.0
        drawdown = p.get("max_drawdown", 0.0)
        p["risk_score"] = min(1.0, concentration_risk * 0.6 + drawdown * 0.4)

        # Recompute goal progress
        target = p.get("target_corpus_inr", 1)
        projected = self._project_corpus(
            total,
            p.get("sip_monthly_inr", 0),
            p.get("investment_horizon_years", 10)
        )
        p["projected_corpus_inr"] = projected
        p["goal_progress"] = min(1.0, projected / max(target, 1))

    def _project_corpus(self, current: float, monthly_sip: float, years: int) -> float:
        months = years * 12
        annual_r = 0.12
        monthly_r = annual_r / 12
        lump_fv = current * ((1 + annual_r) ** years)
        if monthly_r > 0:
            sip_fv = monthly_sip * (((1 + monthly_r) ** months - 1) / monthly_r) * (1 + monthly_r)
        else:
            sip_fv = monthly_sip * months
        return lump_fv + sip_fv

    # ─────────────────────────────────────────────
    # Internal: build observation model
    # ─────────────────────────────────────────────

    def _build_observation(self, last_action_result: Optional[str] = None) -> PortfolioObservation:
        p = self._portfolio
        sector = p.get("sector_exposure", {})
        stocks_raw = p.get("stocks", {})

        return PortfolioObservation(
            cash_inr=round(p["cash_inr"], 2),
            total_portfolio_value_inr=round(p["total_portfolio_value_inr"], 2),
            stocks=StockHoldings(
                RELIANCE=round(stocks_raw.get("RELIANCE", 0), 2),
                TCS=round(stocks_raw.get("TCS", 0), 2),
                INFY=round(stocks_raw.get("INFY", 0), 2),
                HDFCBANK=round(stocks_raw.get("HDFCBANK", 0), 2),
                ICICIBANK=round(stocks_raw.get("ICICIBANK", 0), 2),
                WIPRO=round(stocks_raw.get("WIPRO", 0), 2),
                SUNPHARMA=round(stocks_raw.get("SUNPHARMA", 0), 2),
                HINDUNILVR=round(stocks_raw.get("HINDUNILVR", 0), 2),
            ),
            mutual_fund_value_inr=round(p.get("mutual_fund_value_inr", 0), 2),
            sip_monthly_inr=round(p.get("sip_monthly_inr", 0), 2),
            sector_exposure=SectorExposure(
                IT=round(sector.get("IT", 0), 4),
                Banking=round(sector.get("Banking", 0), 4),
                FMCG=round(sector.get("FMCG", 0), 4),
                Pharma=round(sector.get("Pharma", 0), 4),
                Energy=round(sector.get("Energy", 0), 4),
            ),
            risk_profile=p.get("risk_profile", "moderate"),
            risk_score=round(p.get("risk_score", 0.5), 4),
            investment_horizon_years=p.get("investment_horizon_years", 10),
            target_corpus_inr=p.get("target_corpus_inr", 0),
            goal_progress=round(p.get("goal_progress", 0.0), 4),
            projected_corpus_inr=round(p.get("projected_corpus_inr", 0), 2),
            nifty_trend=p.get("nifty_trend", "sideways"),
            max_drawdown=round(p.get("max_drawdown", 0.0), 4),
            step_number=self._step_count,
            task_name=self.task_name,
            available_actions=VALID_ACTIONS,
            last_action_result=last_action_result,
        )
