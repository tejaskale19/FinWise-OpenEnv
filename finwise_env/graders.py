"""Compatibility wrapper for root deterministic graders."""

from graders import (
    clamp,
    clamp_strict_score,
    linear_score,
    grade_diversify_sector,
    grade_retirement_goal,
    grade_crash_protection,
    compute_step_reward,
    grade_task,
)

__all__ = [
    "clamp",
    "clamp_strict_score",
    "linear_score",
    "grade_diversify_sector",
    "grade_retirement_goal",
    "grade_crash_protection",
    "compute_step_reward",
    "grade_task",
]