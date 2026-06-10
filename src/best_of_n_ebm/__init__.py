"""Synthetic Best-of-N diagnostics for Transformer-structured EBMs."""

from .landscape import CandidateBatch, Prompt, ToyEBMTransformer
from .selection import (
    SelectionResult,
    select_best_of_n,
    select_calibrated_clipped,
    select_diversity_constrained,
)

__all__ = [
    "CandidateBatch",
    "Prompt",
    "SelectionResult",
    "ToyEBMTransformer",
    "select_best_of_n",
    "select_calibrated_clipped",
    "select_diversity_constrained",
]

