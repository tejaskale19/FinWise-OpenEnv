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
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(numeric):
        return lo
    return max(lo, min(hi, numeric))


STRICT_SCORE_MIN = 0.005
STRICT_SCORE_MAX = 0.995
STRICT_SCORE_NONFINITE_FALLBACK = 0.5
ZERO_FALLBACK = float(0)


def _safe_mapping(value: object) -> Dict:
    return value if isinstance(value, dict) else {}


def _safe_number(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def strict_score(score: float) -> float:
    """Universal strict-safe task score clamp with NaN/inf protection."""
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return STRICT_SCORE_NONFINITE_FALLBACK
    if not math.isfinite(numeric):
        return STRICT_SCORE_NONFINITE_FALLBACK
    return max(STRICT_SCORE_MIN, min(STRICT_SCORE_MAX, numeric))


def safe_score(raw_score: float) -> float:
    """Compatibility alias for strict open-interval score clamping."""
    return strict_score(raw_score)


def clamp_strict_score(score: float) -> float:
    """Backward-compatible alias for strict score clamping."""
    return strict_score(score)


def linear_score(current: float, target: float, worst: float) -> float:
    """
    Returns 1.0 when current == target, 0.0 when current == worst.
    Interpolates linearly between.
    """
    if abs(worst - target) < 1e-9:
        return strict_score(1.0)
    score = 1.0 - abs(current - target) / abs(worst - target)
    return strict_score(clamp(score))


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
    state = _safe_mapping(portfolio_state)
    sector = _safe_mapping(state.get("sector_exposure", {}))
    it_weight = _safe_number(sector.get("IT", 0.70), 0.70)

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
    other_sectors = [_safe_number(v, 0.0) for k, v in sector.items() if k != "IT"]
    if other_sectors:
        # Reward balanced distribution (low std deviation across other sectors)
        avg = sum(other_sectors) / len(other_sectors)
        variance = sum((x - avg) ** 2 for x in other_sectors) / len(other_sectors)
        std = math.sqrt(variance)
        diversity_bonus = clamp(0.15 * (1.0 - std / 0.3))
    else:
        diversity_bonus = 0.0

    # Penalty: mutual fund allocation bonus
    mf_value = _safe_number(state.get("mutual_fund_value_inr", 0), 0.0)
    total = _safe_number(state.get("total_portfolio_value_inr", 1), 1.0)
    mf_ratio = mf_value / max(total, 1)
    mf_bonus = clamp(0.05 * (mf_ratio / 0.2))  # up to 0.05 bonus if 20% in MF

    final_score = strict_score(clamp(primary + diversity_bonus + mf_bonus))
    explanation = (
        f"IT exposure={it_weight:.1%} ({label}). "
        f"Diversity bonus={diversity_bonus:.2f}. MF bonus={mf_bonus:.2f}. "
        f"Final score={final_score:.3f}"
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
    current_value = _safe_number(current_value, 0.0)
    monthly_sip = _safe_number(monthly_sip, 0.0)
    years = int(max(0, _safe_number(years, 0.0)))
    annual_return = _safe_number(annual_return, 0.12)

    months = years * 12
    monthly_rate = annual_return / 12

    try:
        # FV of existing lump sum — protect from OverflowError
        if years > 0 and annual_return > 0:
            # Cap exponent to prevent overflow: (1.12 ** 500) already overflows
            if years > 300:
                return ZERO_FALLBACK
            lump_fv = current_value * ((1 + annual_return) ** years)
            if not math.isfinite(lump_fv):
                return ZERO_FALLBACK
        else:
            lump_fv = current_value

        # FV of SIP stream
        if monthly_rate > 0:
            if months > 5000:  # Prevent SIP formula overflow
                return ZERO_FALLBACK
            sip_fv = monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
            if not math.isfinite(sip_fv):
                return ZERO_FALLBACK
        else:
            sip_fv = monthly_sip * months

        result = lump_fv + sip_fv
        if not math.isfinite(result):
            return ZERO_FALLBACK
        return result
    except (OverflowError, ValueError):
        # Overflow or computation error — return safe default
        return ZERO_FALLBACK


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
    state = _safe_mapping(portfolio_state)
    total_value = _safe_number(state.get("total_portfolio_value_inr", 0), 0.0)
    sip = _safe_number(state.get("sip_monthly_inr", 3000), 3000.0)
    horizon = int(max(0, _safe_number(state.get("investment_horizon_years", 15), 15.0)))
    target = _safe_number(state.get("target_corpus_inr", 1_00_00_000), 1_00_00_000.0)
    mf_value = _safe_number(state.get("mutual_fund_value_inr", 0), 0.0)

    projected = _project_corpus(total_value, sip, horizon)
    if not math.isfinite(projected):
        projected = 0.0
    ratio = projected / max(target, 1)
    ratio = clamp(ratio, 0.0, 10.0)

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
    sector = _safe_mapping(state.get("sector_exposure", {}))
    it = _safe_number(sector.get("IT", 1.0), 1.0)
    diversification_bonus = 0.05 if it < 0.35 else 0.0

    final_score = strict_score(clamp(primary + sip_bonus + mf_bonus + diversification_bonus))
    explanation = (
        f"Projected corpus=₹{projected:,.0f} vs target=₹{target:,.0f} ({ratio:.1%}). "
        f"SIP=₹{sip:,}/mo (bonus={sip_bonus:.2f}). "
        f"MF bonus={mf_bonus:.2f}. Diversity bonus={diversification_bonus:.2f}. "
        f"Final score={final_score:.3f}"
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
    state = _safe_mapping(portfolio_state)
    drawdown = _safe_number(state.get("max_drawdown", 0.18), 0.18)
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
    cash = _safe_number(state.get("cash_inr", 25_000), 25_000.0)
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
    sector = _safe_mapping(state.get("sector_exposure", {}))
    banking = _safe_number(sector.get("Banking", 0.45), 0.45)
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
    final_score = strict_score(clamp(
        preservation_score * 0.40
        + liquidity_score * 0.35
        + banking_score * 0.25
    ))

    explanation = (
        f"Capital preservation: drawdown={drawdown:.1%} → score={preservation_score:.2f} (w=0.40). "
        f"Liquidity: cash=₹{cash:,} → score={liquidity_score:.2f} (w=0.35). "
        f"Banking reduction: {banking:.1%} → score={banking_score:.2f} (w=0.25). "
        f"Final score={final_score:.3f}"
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
    prev_state = _safe_mapping(prev_state)
    curr_state = _safe_mapping(curr_state)
    prev_sector = _safe_mapping(prev_state.get("sector_exposure", {}))
    curr_sector = _safe_mapping(curr_state.get("sector_exposure", {}))

    prev_it = _safe_number(prev_sector.get("IT", 0.5), 0.5)
    curr_it = _safe_number(curr_sector.get("IT", 0.5), 0.5)
    it_reduction = prev_it - curr_it  # positive = improvement
    diversification_delta = clamp(it_reduction * 2.0, -0.5, 0.5)

    # ── corpus growth ──
    prev_progress = _safe_number(prev_state.get("goal_progress", 0.0), 0.0)
    curr_progress = _safe_number(curr_state.get("goal_progress", 0.0), 0.0)
    corpus_growth_delta = clamp((curr_progress - prev_progress) * 5.0, -0.5, 0.5)

    # ── risk reduction ──
    prev_risk = _safe_number(prev_state.get("risk_score", 0.5), 0.5)
    curr_risk = _safe_number(curr_state.get("risk_score", 0.5), 0.5)
    risk_reduction_delta = clamp((prev_risk - curr_risk) * 2.0, -0.3, 0.3)

    # ── liquidity improvement ──
    prev_cash = _safe_number(prev_state.get("cash_inr", 0), 0.0)
    curr_cash = _safe_number(curr_state.get("cash_inr", 0), 0.0)
    liquidity_delta = clamp((curr_cash - prev_cash) / 2_00_000, -0.3, 0.3)

    # ── penalties ──
    # Overconcentration: penalise if any sector > 60%
    sector = curr_sector
    sector_values = [_safe_number(v, 0.0) for v in sector.values()]
    max_sector = max(sector_values) if sector_values else 0.0
    overconcentration_penalty = 0.15 if max_sector > 0.60 else (0.05 if max_sector > 0.50 else 0.0)

    # Churn: penalise selling AND buying same asset class in same step
    churn_penalty = 0.05 if action_type in ("sell_stock", "buy_stock") else 0.0

    # Cash depletion: penalise going below ₹20,000 cash
    current_cash_for_penalty = _safe_number(curr_state.get("cash_inr", 0), 0.0)
    cash_depletion_penalty = 0.10 if current_cash_for_penalty < 20_000 else 0.0

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
    # !! CRITICAL: Clamp reward to STRICT bounds (0, 1), not just [-1, 1]
    reward = strict_score(reward)

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
    state = _safe_mapping(portfolio_state)
    if task_name == "diversify_sector_easy":
        score, explanation = grade_diversify_sector(state)
    elif task_name == "retirement_goal_medium":
        score, explanation = grade_retirement_goal(state)
    elif task_name == "crash_protection_hard":
        score, explanation = grade_crash_protection(state)
    else:
        score = STRICT_SCORE_NONFINITE_FALLBACK
        explanation = f"Unknown task: {task_name}. Applied fallback score."

    score = strict_score(score)
    return score, explanation
