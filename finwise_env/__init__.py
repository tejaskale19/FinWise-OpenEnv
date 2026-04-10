"""Compatibility package exposing FinWise OpenEnv symbols."""

from finwise_env.models import (
    PortfolioObservation,
    PortfolioAction,
    RewardBreakdown,
    StepResult,
    SectorExposure,
    StockHoldings,
    TaskDefinition,
)

__all__ = [
    "FinWiseEnv",
    "PortfolioObservation",
    "PortfolioAction",
    "RewardBreakdown",
    "StepResult",
    "SectorExposure",
    "StockHoldings",
    "TaskDefinition",
]


def __getattr__(name: str):
    if name == "FinWiseEnv":
        from finwise_env.env import FinWiseEnv
        return FinWiseEnv
    raise AttributeError(f"module 'finwise_env' has no attribute '{name}'")