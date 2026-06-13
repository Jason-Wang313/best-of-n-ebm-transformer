from __future__ import annotations

import numpy as np

from .landscape import CandidateBatch
from .selection import SelectionResult


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())


def spearman_like(x: np.ndarray, y: np.ndarray) -> float:
    """Small dependency-free rank correlation for diagnostics."""

    if x.size < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def pool_diagnostics(batch: CandidateBatch, selected: SelectionResult) -> dict[str, float]:
    energy = batch.proxy_energy
    true = batch.true_score
    q10 = float(np.quantile(energy, 0.10))
    low_energy = energy <= q10
    invalid_low_energy_rate = float((~batch.valid[low_energy]).mean()) if low_energy.any() else 0.0

    signatures = np.array(["-".join(map(str, row[:4])) for row in batch.tokens])
    selected_signature = "-".join(map(str, batch.tokens[selected.index, :4]))

    return {
        "candidate_count": float(energy.shape[0]),
        "selected_energy": float(selected.proxy_energy),
        "selected_true_score": float(selected.true_score),
        "selected_valid": float(selected.valid),
        "selected_shortcut": float(selected.attention_shortcut),
        "energy_tail_gap": float(np.median(energy) - selected.proxy_energy),
        "invalid_low_energy_rate": invalid_low_energy_rate,
        "pool_valid_rate": float(batch.valid.mean()),
        "pool_true_score": float(true.mean()),
        "score_execution_spearman": spearman_like(-energy, true),
        "mode_entropy": _entropy(signatures),
        "selected_mode_frequency": float((signatures == selected_signature).mean()),
        "artifact_selected": float(selected.proposal_type == "artifact"),
    }


def aggregate_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([row[key] for row in rows], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_stderr"] = float(vals.std(ddof=1) / np.sqrt(max(1, vals.size)))
    return out

