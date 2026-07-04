"""Simple, non-model forecasting baselines."""

from baselines.naive import (
    BASELINES,
    BaselineContext,
    BaselineOutput,
    prepare_context,
)

__all__ = ["BASELINES", "BaselineContext", "BaselineOutput", "prepare_context"]
