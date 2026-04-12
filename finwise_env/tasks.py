"""
FinWise OpenEnv — Task Definitions
3 tasks with clear difficulty progression: easy → medium → hard
"""

from __future__ import annotations
from typing import Dict
from models import TaskDefinition


# ─────────────────────────────────────────────────────────────
# TASK 1 — EASY: Sector Diversification
# Agent must reduce IT sector overexposure from 70% → <40%
# ─────────────────────────────────────────────────────────────

TASK_DIVERSIFY_EASY = TaskDefinition(
    name="diversify_sector_easy",
    difficulty="easy",
    description=(
        "An aggressive investor has 70% of their portfolio concentrated in IT sector stocks. "
        "Reduce IT sector exposure to below 40% by selling IT stocks and buying into other sectors "
        "or mutual funds. The portfolio has ₹5,00,000 total value with ₹50,000 cash available."
    ),
    max_steps=8,
    success_threshold=0.8,
    initial_portfolio={
        "cash_inr": 50_000,
        "total_portfolio_value_inr": 5_00_000,
        "stocks": {
            "RELIANCE": 15_000,
            "TCS": 1_50_000,
            "INFY": 1_45_000,
            "HDFCBANK": 30_000,
            "ICICIBANK": 30_000,
            "WIPRO": 55_000,
            "SUNPHARMA": 10_000,
            "HINDUNILVR": 15_000,
        },
        "mutual_fund_value_inr": 0,
        "sip_monthly_inr": 5_000,
        "sector_exposure": {
            "IT": 0.70,
            "Banking": 0.12,
            "FMCG": 0.03,
            "Pharma": 0.02,
            "Energy": 0.03,
        },
        "risk_profile": "aggressive",
        "risk_score": 0.82,
        "investment_horizon_years": 10,
        "target_corpus_inr": 25_00_000,
        "goal_progress": 0.20,
        "projected_corpus_inr": 9_00_000,
        "nifty_trend": "bullish",
        "max_drawdown": 0.05,
    }
)


# ─────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Retirement Corpus Optimization
# Agent must improve goal achievement probability for a 25-year-old
# ─────────────────────────────────────────────────────────────

TASK_RETIREMENT_MEDIUM = TaskDefinition(
    name="retirement_goal_medium",
    difficulty="medium",
    description=(
        "A 25-year-old moderate investor wants to build a retirement corpus of ₹1 Crore in 15 years. "
        "Current monthly SIP of ₹3,000 is insufficient. Portfolio is also underweighted in equity. "
        "The agent must increase SIP, shift allocation toward equity mutual funds, and ensure "
        "proper diversification across sectors to maximize the probability of reaching the target corpus. "
        "Market is currently sideways with moderate volatility."
    ),
    max_steps=10,
    success_threshold=0.75,
    initial_portfolio={
        "cash_inr": 80_000,
        "total_portfolio_value_inr": 3_50_000,
        "stocks": {
            "RELIANCE": 40_000,
            "TCS": 30_000,
            "INFY": 25_000,
            "HDFCBANK": 50_000,
            "ICICIBANK": 35_000,
            "WIPRO": 20_000,
            "SUNPHARMA": 25_000,
            "HINDUNILVR": 45_000,
        },
        "mutual_fund_value_inr": 0,
        "sip_monthly_inr": 3_000,
        "sector_exposure": {
            "IT": 0.21,
            "Banking": 0.24,
            "FMCG": 0.13,
            "Pharma": 0.07,
            "Energy": 0.11,
        },
        "risk_profile": "moderate",
        "risk_score": 0.48,
        "investment_horizon_years": 15,
        "target_corpus_inr": 1_00_00_000,
        "goal_progress": 0.04,
        "projected_corpus_inr": 18_00_000,
        "nifty_trend": "sideways",
        "max_drawdown": 0.08,
    }
)


# ─────────────────────────────────────────────────────────────
# TASK 3 — HARD: Crash Protection + Liquidity Management
# Banking sector crash, investor needs liquidity in 6 months
# ─────────────────────────────────────────────────────────────

TASK_CRASH_HARD = TaskDefinition(
    name="crash_protection_hard",
    difficulty="hard",
    description=(
        "CRISIS SCENARIO: Banking sector crash is underway (-22% in 30 days). "
        "An aggressive investor has 45% exposure to banking stocks (HDFCBANK, ICICIBANK). "
        "Portfolio drawdown is already 18%. The investor needs ₹2,00,000 liquid cash within 6 months "
        "for a home loan down payment. "
        "The agent must: (1) reduce banking sector exposure, (2) preserve capital, "
        "(3) build adequate liquidity, and (4) maintain some growth allocation for long-term goals. "
        "This requires balancing conflicting objectives under severe market stress. "
        "Selling during crash locks in losses but prevents further damage."
    ),
    max_steps=12,
    success_threshold=0.70,
    initial_portfolio={
        "cash_inr": 25_000,
        "total_portfolio_value_inr": 8_20_000,
        "stocks": {
            "RELIANCE": 80_000,
            "TCS": 60_000,
            "INFY": 50_000,
            "HDFCBANK": 2_10_000,  # overweight — crashing
            "ICICIBANK": 1_60_000,  # overweight — crashing
            "WIPRO": 40_000,
            "SUNPHARMA": 50_000,
            "HINDUNILVR": 45_000,
        },
        "mutual_fund_value_inr": 0,
        "sip_monthly_inr": 15_000,
        "sector_exposure": {
            "IT": 0.18,
            "Banking": 0.45,  # dangerously high
            "FMCG": 0.05,
            "Pharma": 0.06,
            "Energy": 0.10,
        },
        "risk_profile": "aggressive",
        "risk_score": 0.91,
        "investment_horizon_years": 8,
        "target_corpus_inr": 50_00_000,
        "goal_progress": 0.16,
        "projected_corpus_inr": 26_00_000,
        "nifty_trend": "bearish",
        "max_drawdown": 0.18,  # already 18% down
    }
)


# ─────────────────────────────────────────────────────────────
# Registry — all tasks indexed by name
# ─────────────────────────────────────────────────────────────

TASK_REGISTRY: Dict[str, TaskDefinition] = {
    TASK_DIVERSIFY_EASY.name: TASK_DIVERSIFY_EASY,
    TASK_RETIREMENT_MEDIUM.name: TASK_RETIREMENT_MEDIUM,
    TASK_CRASH_HARD.name: TASK_CRASH_HARD,
}

ALL_TASK_NAMES = list(TASK_REGISTRY.keys())
