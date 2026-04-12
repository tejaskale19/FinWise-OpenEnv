"""
FinWise OpenEnv — Typed Pydantic Models
Defines Observation, Action, and Reward models for the portfolio advisory environment.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# OBSERVATION — what the LLM agent sees
# ─────────────────────────────────────────────

class SectorExposure(BaseModel):
    IT: float = Field(default=0.0, ge=0.0, le=1.0, description="IT sector weight (0–1)")
    Banking: float = Field(default=0.0, ge=0.0, le=1.0, description="Banking sector weight (0–1)")
    FMCG: float = Field(default=0.0, ge=0.0, le=1.0, description="FMCG sector weight (0–1)")
    Pharma: float = Field(default=0.0, ge=0.0, le=1.0, description="Pharma sector weight (0–1)")
    Energy: float = Field(default=0.0, ge=0.0, le=1.0, description="Energy sector weight (0–1)")


class StockHoldings(BaseModel):
    RELIANCE: float = Field(default=0.0, ge=0.0, description="Reliance value in INR")
    TCS: float = Field(default=0.0, ge=0.0, description="TCS value in INR")
    INFY: float = Field(default=0.0, ge=0.0, description="Infosys value in INR")
    HDFCBANK: float = Field(default=0.0, ge=0.0, description="HDFC Bank value in INR")
    ICICIBANK: float = Field(default=0.0, ge=0.0, description="ICICI Bank value in INR")
    WIPRO: float = Field(default=0.0, ge=0.0, description="Wipro value in INR")
    SUNPHARMA: float = Field(default=0.0, ge=0.0, description="Sun Pharma value in INR")
    HINDUNILVR: float = Field(default=0.0, ge=0.0, description="HUL value in INR")


class PortfolioObservation(BaseModel):
    """Complete portfolio snapshot — what the agent receives each step."""

    # Core financials
    cash_inr: float = Field(description="Available cash in INR")
    total_portfolio_value_inr: float = Field(description="Total portfolio value in INR")

    # Holdings
    stocks: StockHoldings = Field(description="Individual stock values in INR")
    mutual_fund_value_inr: float = Field(default=0.0, description="Total mutual fund value")
    sip_monthly_inr: float = Field(description="Monthly SIP amount in INR")

    # Risk and allocation
    sector_exposure: SectorExposure = Field(description="Sector-wise allocation (weights 0–1)")
    risk_profile: str = Field(description="Investor risk profile: conservative/moderate/aggressive")
    risk_score: float = Field(ge=0.0, le=1.0, description="Current portfolio risk score (0=low, 1=high)")

    # Goal tracking
    investment_horizon_years: int = Field(description="Years remaining to goal")
    target_corpus_inr: float = Field(description="Target corpus amount in INR")
    goal_progress: float = Field(ge=0.0, le=1.0, description="Progress toward target corpus (0–1)")
    projected_corpus_inr: float = Field(description="Projected corpus at horizon with current allocation")

    # Market context
    nifty_trend: str = Field(description="NIFTY trend: bullish/bearish/sideways")
    max_drawdown: float = Field(ge=0.0, le=1.0, description="Maximum drawdown of portfolio (0–1)")
    step_number: int = Field(description="Current step in episode")
    task_name: str = Field(description="Active task identifier")
    available_actions: List[str] = Field(description="List of valid action types for this step")
    last_action_result: Optional[str] = Field(default=None, description="Result message from last action")


# ─────────────────────────────────────────────
# ACTION — what the LLM agent can do
# ─────────────────────────────────────────────

class PortfolioAction(BaseModel):
    """
    Action the agent takes. Provide action_type and relevant parameters.

    action_type options:
      - buy_stock       : asset=STOCK_SYMBOL, amount_inr=float
      - sell_stock      : asset=STOCK_SYMBOL, amount_inr=float
      - increase_sip    : amount_inr=float (monthly increase)
      - decrease_sip    : amount_inr=float (monthly decrease)
      - buy_mutual_fund : amount_inr=float
      - sell_mutual_fund: amount_inr=float
      - rebalance_sector: asset=SECTOR_NAME, target_weight=float (0–1)
      - hold            : no parameters needed
    """
    action_type: str = Field(
        description="Type of action to perform",
        examples=["buy_stock", "sell_stock", "increase_sip", "buy_mutual_fund",
                  "sell_mutual_fund", "rebalance_sector", "hold"]
    )
    asset: Optional[str] = Field(
        default=None,
        description="Stock symbol (e.g. TCS, INFY) or sector name (e.g. IT, Banking)"
    )
    amount_inr: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Amount in INR for buy/sell/SIP actions"
    )
    target_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Target sector weight for rebalance_sector action (0–1)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's explanation for this action (used for explainability scoring)"
    )


# ─────────────────────────────────────────────
# REWARD — structured reward breakdown
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Detailed reward signal returned after each step."""

    total_reward: float = Field(description="Composite reward for this step (-1 to 1)")

    # Component rewards
    diversification_delta: float = Field(description="Change in diversification score")
    corpus_growth_delta: float = Field(description="Change in projected corpus progress")
    risk_reduction_delta: float = Field(description="Change in portfolio risk score (positive = risk reduced)")
    liquidity_safety_delta: float = Field(description="Change in cash/liquidity safety margin")

    # Penalties applied
    overconcentration_penalty: float = Field(default=0.0, description="Penalty for high sector concentration")
    churn_penalty: float = Field(default=0.0, description="Penalty for excessive buy/sell churn")
    cash_depletion_penalty: float = Field(default=0.0, description="Penalty for reducing cash below safety threshold")

    explanation: str = Field(description="Human-readable reason for this reward")


# ─────────────────────────────────────────────
# STEP RESULT — returned by env.step()
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """Full result returned from env.step(action)."""
    observation: PortfolioObservation
    reward: float = Field(description="Scalar reward for this step")
    reward_breakdown: RewardBreakdown
    done: bool = Field(description="True if episode is complete")
    info: Dict = Field(default_factory=dict, description="Extra diagnostics")


# ─────────────────────────────────────────────
# TASK DEFINITION
# ─────────────────────────────────────────────

class TaskDefinition(BaseModel):
    """Metadata for each task in the environment."""
    name: str
    difficulty: str  # easy / medium / hard
    description: str
    max_steps: int
    success_threshold: float = Field(ge=0.0, le=1.0)
    initial_portfolio: Dict
