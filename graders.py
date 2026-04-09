"""
FinWise OpenEnv — Deterministic Graders
Each grader returns a score in [0.0, 1.0] with partial progress signals.
Graders are reproducible and deterministic given the same portfolio state.
"""

from __future__ import annotations
from typing import Dict, Tuple
import math


# ─────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def linear_score(current: float, target: float, worst: float) -> float:
    """
    Returns 1.0 when current == target, 0.0 when current == worst.
    Interpolates linearly between.
    """
    if abs(worst - target) < 1e-9:
        return 1.0
    score = 1.0 - abs(current - target) / abs(worst - target)
    return clamp(score)


# ─────────────────────────────────────────────────────────────
# GRADER 1 — EASY: Sector Diversification
# Goal: reduce IT exposure from ~70% to below 40%
# ─────────────────────────────────────────────────────────────

def grade_diversify_sector(portfolio_state: Dict) -> Tuple[float, str]:
    """
    Score based on how well the agent diversified the IT sector.

    Scoring:
      IT ≤ 0.30  → 1.00 (excellent)
      IT ≤ 0.35  → 0.90
      IT ≤ 0.40  → 0.80 (minimum success)
      IT ≤ 0.50  → 0.55 (partial progress)
      IT ≤ 0.60  → 0.30
      IT > 0.60  → 0.10 (barely any progress)

    Also rewards secondary diversification across sectors.
    """
    sector = portfolio_state.get("sector_exposure", {})
    it_weight = sector.get("IT", 0.70)

    # Primary score: IT reduction
    if it_weight <= 0.30:
        primary = 1.00
        label = "excellent"
    elif it_weight <= 0.35:
        primary = 0.90
        label = "very good"
    elif it_weight <= 0.40:
        primary = 0.80
        label = "success"
    elif it_weight <= 0.50:
        primary = 0.55
        label = "partial"
    elif it_weight <= 0.60:
        primary = 0.30
        label = "minimal"
    else:
        primary = 0.10
        label = "no progress"

    # Secondary score: distribution across other sectors
    other_sectors = [v for k, v in sector.items() if k != "IT"]
    if other_sectors:
        # Reward balanced distribution (low std deviation across other sectors)
        avg = sum(other_sectors) / len(other_sectors)
        variance = sum((x - avg) ** 2 for x in other_sectors) / len(other_sectors)
        std = math.sqrt(variance)
        diversity_bonus = clamp(0.15 * (1.0 - std / 0.3))
    else:
        diversity_bonus = 0.0

    # Penalty: mutual fund allocation bonus
    mf_value = portfolio_state.get("mutual_fund_value_inr", 0)
    total = portfolio_state.get("total_portfolio_value_inr", 1)
    mf_ratio = mf_value / max(total, 1)
    mf_bonus = clamp(0.05 * (mf_ratio / 0.2))  # up to 0.05 bonus if 20% in MF

    final_score = clamp(primary + diversity_bonus + mf_bonus)
    final_score = max(0.01, min(0.99, final_score))
    explanation = (
        f"IT exposure={it_weight:.1%} ({label}). "
        f"Diversity bonus={diversity_bonus:.2f}. MF bonus={mf_bonus:.2f}. "
        f"Final score={final_score:.2f}"
    )
    return final_score, explanation


# ─────────────────────────────────────────────────────────────
# GRADER 2 — MEDIUM: Retirement Corpus Optimization
# Goal: improve probability of reaching ₹1Cr in 15 years
# ─────────────────────────────────────────────────────────────

def _project_corpus(
    current_value: float,
    monthly_sip: float,
    years: int,
    annual_return: float = 0.12
) -> float:
    """Project final corpus using standard SIP formula."""
    months = years * 12
    monthly_rate = annual_return / 12

    # FV of existing lump sum
    lump_fv = current_value * ((1 + annual_return) ** years)

    # FV of SIP stream
    if monthly_rate > 0:
        sip_fv = monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        sip_fv = monthly_sip * months

    return lump_fv + sip_fv


def grade_retirement_goal(portfolio_state: Dict) -> Tuple[float, str]:
    """
    Score based on improvement in projected corpus vs target.

    Scoring:
      projected ≥ 100% of target → 1.00
      projected ≥  80% of target → 0.85
      projected ≥  60% of target → 0.70
      projected ≥  40% of target → 0.50
      projected ≥  25% of target → 0.30
      else                       → 0.10

    Bonuses for:
      - SIP increase (up to 0.10)
      - Adding mutual funds (up to 0.05)
      - Staying diversified (up to 0.05)
    """
    total_value = portfolio_state.get("total_portfolio_value_inr", 0)
    sip = portfolio_state.get("sip_monthly_inr", 3000)
    horizon = portfolio_state.get("investment_horizon_years", 15)
    target = portfolio_state.get("target_corpus_inr", 1_00_00_000)
    mf_value = portfolio_state.get("mutual_fund_value_inr", 0)

    projected = _project_corpus(total_value, sip, horizon)
    ratio = projected / max(target, 1)

    if ratio >= 1.0:
        primary = 1.00
    elif ratio >= 0.80:
        primary = 0.85
    elif ratio >= 0.60:
        primary = 0.70
    elif ratio >= 0.40:
        primary = 0.50
    elif ratio >= 0.25:
        primary = 0.30
    else:
        primary = 0.10

    # SIP bonus: reward increase above baseline ₹3000
    sip_bonus = clamp(0.10 * min((sip - 3000) / 12000, 1.0)) if sip > 3000 else 0.0

    # MF bonus
    mf_ratio = mf_value / max(total_value, 1)
    mf_bonus = clamp(0.05 * min(mf_ratio / 0.3, 1.0))

    # Diversification bonus
    sector = portfolio_state.get("sector_exposure", {})
    it = sector.get("IT", 1.0)
    diversification_bonus = 0.05 if it < 0.35 else 0.0

    final_score = clamp(primary + sip_bonus + mf_bonus + diversification_bonus)
    final_score = max(0.01, min(0.99, final_score))
    explanation = (
        f"Projected corpus=₹{projected:,.0f} vs target=₹{target:,.0f} ({ratio:.1%}). "
        f"SIP=₹{sip:,}/mo (bonus={sip_bonus:.2f}). "
        f"MF bonus={mf_bonus:.2f}. Diversity bonus={diversification_bonus:.2f}. "
        f"Final score={final_score:.2f}"
    )
    return final_score, explanation


# ─────────────────────────────────────────────────────────────
# GRADER 3 — HARD: Crash Protection + Liquidity
# Goal: preserve capital + build liquidity + maintain growth
# ─────────────────────────────────────────────────────────────

def grade_crash_protection(portfolio_state: Dict) -> Tuple[float, str]:
    """
    Multi-objective grader for the crisis scenario.

    Three sub-scores (weighted):
      1. Capital preservation (40%): reduce max_drawdown below 15%
      2. Liquidity safety (35%): cash ≥ ₹2,00,000 needed for home loan
      3. Banking exposure reduction (25%): banking sector < 25%

    Each sub-score: 0.0–1.0 with partial progress.
    """
    # 1. Capital preservation: reward stopping the bleeding
    drawdown = portfolio_state.get("max_drawdown", 0.18)
    if drawdown <= 0.08:
        preservation_score = 1.0
    elif drawdown <= 0.12:
        preservation_score = 0.80
    elif drawdown <= 0.15:
        preservation_score = 0.60
    elif drawdown <= 0.18:
        preservation_score = 0.35
    else:
        preservation_score = max(0.0, 1.0 - (drawdown / 0.40))

    # 2. Liquidity: need ₹2,00,000 cash for home loan
    cash = portfolio_state.get("cash_inr", 25_000)
    liquidity_target = 2_00_000
    if cash >= liquidity_target:
        liquidity_score = 1.0
    elif cash >= 1_50_000:
        liquidity_score = 0.75
    elif cash >= 1_00_000:
        liquidity_score = 0.50
    elif cash >= 50_000:
        liquidity_score = 0.25
    else:
        liquidity_score = clamp(cash / liquidity_target)

    # 3. Banking sector reduction: from 45% → target <25%
    sector = portfolio_state.get("sector_exposure", {})
    banking = sector.get("Banking", 0.45)
    if banking <= 0.20:
        banking_score = 1.0
    elif banking <= 0.25:
        banking_score = 0.85
    elif banking <= 0.30:
        banking_score = 0.65
    elif banking <= 0.38:
        banking_score = 0.40
    else:
        banking_score = max(0.0, 1.0 - (banking - 0.25) / 0.20)

    # Weighted composite
    final_score = clamp(
        preservation_score * 0.40
        + liquidity_score * 0.35
        + banking_score * 0.25
    )
    final_score = max(0.01, min(0.99, final_score))

    explanation = (
        f"Capital preservation: drawdown={drawdown:.1%} → score={preservation_score:.2f} (w=0.40). "
        f"Liquidity: cash=₹{cash:,} → score={liquidity_score:.2f} (w=0.35). "
        f"Banking reduction: {banking:.1%} → score={banking_score:.2f} (w=0.25). "
        f"Final score={final_score:.2f}"
    )
    return final_score, explanation


# ─────────────────────────────────────────────────────────────
# REWARD FUNCTION — step-level signal (not just terminal)
# ─────────────────────────────────────────────────────────────

def compute_step_reward(
    prev_state: Dict,
    curr_state: Dict,
    action_type: str,
    task_name: str
) -> Tuple[float, Dict]:
    """
    Compute step-level reward showing trajectory progress.
    Returns (scalar_reward, breakdown_dict).

    Formula:
      reward = diversification_delta * 0.30
             + corpus_growth_delta  * 0.30
             + risk_reduction_delta * 0.20
             + liquidity_delta      * 0.20
             - overconcentration_penalty
             - churn_penalty
             - cash_depletion_penalty
    """
    # ── diversification improvement ──
    prev_it = prev_state.get("sector_exposure", {}).get("IT", 0.5)
    curr_it = curr_state.get("sector_exposure", {}).get("IT", 0.5)
    it_reduction = prev_it - curr_it  # positive = improvement
    diversification_delta = clamp(it_reduction * 2.0, -0.5, 0.5)

    # ── corpus growth ──
    prev_progress = prev_state.get("goal_progress", 0.0)
    curr_progress = curr_state.get("goal_progress", 0.0)
    corpus_growth_delta = clamp((curr_progress - prev_progress) * 5.0, -0.5, 0.5)

    # ── risk reduction ──
    prev_risk = prev_state.get("risk_score", 0.5)
    curr_risk = curr_state.get("risk_score", 0.5)
    risk_reduction_delta = clamp((prev_risk - curr_risk) * 2.0, -0.3, 0.3)

    # ── liquidity improvement ──
    prev_cash = prev_state.get("cash_inr", 0)
    curr_cash = curr_state.get("cash_inr", 0)
    liquidity_delta = clamp((curr_cash - prev_cash) / 2_00_000, -0.3, 0.3)

    # ── penalties ──
    # Overconcentration: penalise if any sector > 60%
    sector = curr_state.get("sector_exposure", {})
    max_sector = max(sector.values()) if sector else 0.0
    overconcentration_penalty = 0.15 if max_sector > 0.60 else (0.05 if max_sector > 0.50 else 0.0)

    # Churn: penalise selling AND buying same asset class in same step
    churn_penalty = 0.05 if action_type in ("sell_stock", "buy_stock") else 0.0

    # Cash depletion: penalise going below ₹20,000 cash
    cash_depletion_penalty = 0.10 if curr_state.get("cash_inr", 0) < 20_000 else 0.0

    # ── composite reward ──
    reward = (
        diversification_delta * 0.30
        + corpus_growth_delta * 0.30
        + risk_reduction_delta * 0.20
        + liquidity_delta * 0.20
        - overconcentration_penalty
        - churn_penalty
        - cash_depletion_penalty
    )
    reward = clamp(reward, -1.0, 1.0)

    breakdown = {
        "diversification_delta": round(diversification_delta, 3),
        "corpus_growth_delta": round(corpus_growth_delta, 3),
        "risk_reduction_delta": round(risk_reduction_delta, 3),
        "liquidity_safety_delta": round(liquidity_delta, 3),
        "overconcentration_penalty": round(overconcentration_penalty, 3),
        "churn_penalty": round(churn_penalty, 3),
        "cash_depletion_penalty": round(cash_depletion_penalty, 3),
        "total_reward": round(reward, 3),
    }
    return reward, breakdown


# ─────────────────────────────────────────────────────────────
# GRADE DISPATCHER — call correct grader by task name
# ─────────────────────────────────────────────────────────────

def grade_task(task_name: str, portfolio_state: Dict) -> Tuple[float, str]:
    if task_name == "diversify_sector_easy":
        score, explanation = grade_diversify_sector(portfolio_state)
    elif task_name == "retirement_goal_medium":
        score, explanation = grade_retirement_goal(portfolio_state)
    elif task_name == "crash_protection_hard":
        score, explanation = grade_crash_protection(portfolio_state)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    score = max(0.01, min(0.99, float(score)))
    return score, explanation
