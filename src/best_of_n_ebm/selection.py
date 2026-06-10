from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landscape import CandidateBatch


@dataclass(frozen=True)
class SelectionResult:
    method: str
    index: int
    score_used: float
    proxy_energy: float
    true_score: float
    valid: bool
    attention_shortcut: float
    repetition: float
    proposal_type: str


def _result(method: str, batch: CandidateBatch, idx: int, score_used: float) -> SelectionResult:
    return SelectionResult(
        method=method,
        index=int(idx),
        score_used=float(score_used),
        proxy_energy=float(batch.proxy_energy[idx]),
        true_score=float(batch.true_score[idx]),
        valid=bool(batch.valid[idx]),
        attention_shortcut=float(batch.attention_shortcut[idx]),
        repetition=float(batch.repetition[idx]),
        proposal_type=str(batch.proposal_type[idx]),
    )


def select_best_of_n(batch: CandidateBatch) -> SelectionResult:
    idx = int(np.argmin(batch.proxy_energy))
    return _result("bon", batch, idx, batch.proxy_energy[idx])


def select_calibrated_clipped(
    batch: CandidateBatch,
    clip_quantile: float = 0.20,
    shortcut_weight: float = 1.00,
) -> SelectionResult:
    """Select with tail clipping and an attention-shortcut penalty.

    The repair uses only proxy-side quantities available from the energy model:
    candidate energies and their decomposition into attention-shortcut mass. It
    does not inspect true validity or task labels.
    """

    lower = float(np.quantile(batch.proxy_energy, clip_quantile))
    shortcut_center = float(np.median(batch.attention_shortcut))
    clipped = np.maximum(batch.proxy_energy, lower)
    shortcut_excess = np.maximum(0.0, batch.attention_shortcut - shortcut_center)
    score = clipped + shortcut_weight * shortcut_excess
    idx = int(np.argmin(score))
    return _result("calibrated_clipped", batch, idx, score[idx])


def select_diversity_constrained(
    batch: CandidateBatch,
    energy_quantile: float = 0.20,
    repetition_weight: float = 0.20,
) -> SelectionResult:
    cutoff = float(np.quantile(batch.proxy_energy, energy_quantile))
    eligible = np.flatnonzero(batch.proxy_energy <= cutoff)
    if eligible.size == 0:
        eligible = np.arange(batch.proxy_energy.shape[0])
    score = batch.proxy_energy[eligible] + repetition_weight * batch.repetition[eligible]
    idx = int(eligible[int(np.argmin(score))])
    return _result("diversity_constrained", batch, idx, score.min())
