from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClaimStatus:
    claim: str
    status: str
    evidence: str


def evaluate_claims(summary_rows: list[dict[str, object]]) -> list[ClaimStatus]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    bon = sorted(by_method.get("bon", []), key=lambda r: int(r["N"]))
    repaired = sorted(by_method.get("calibrated_clipped", []), key=lambda r: int(r["N"]))
    claims: list[ClaimStatus] = []

    if bon:
        n_values = [int(r["N"]) for r in bon]
        energies = np.array([float(r["selected_energy_mean"]) for r in bon])
        validity = np.array([float(r["selected_valid_mean"]) for r in bon])
        true_score = np.array([float(r["selected_true_score_mean"]) for r in bon])
        energy_drop = energies[0] - energies[-1]
        validity_drop = validity[0] - validity[-1]
        true_drop = true_score[0] - true_score[-1]
        status = "supported" if energy_drop > 0.25 and validity_drop > 0.25 else "weak"
        claims.append(
            ClaimStatus(
                "Best-of-N drives selected candidates into lower proxy-energy tails.",
                "supported" if energy_drop > 0.25 else "weak",
                f"N={n_values[0]} to N={n_values[-1]} selected energy changed by {-energy_drop:.3f}.",
            )
        )
        claims.append(
            ClaimStatus(
                "Proxy-energy tail exploitation reduces semantic validity.",
                status,
                f"Validity changed by {-validity_drop:.3f}; true score changed by {-true_drop:.3f}.",
            )
        )

    if bon and repaired:
        bon_last = bon[-1]
        rep_last = repaired[-1]
        repair_gain = float(rep_last["selected_true_score_mean"]) - float(
            bon_last["selected_true_score_mean"]
        )
        repair_energy = float(rep_last["selected_energy_mean"]) - float(
            bon_last["selected_energy_mean"]
        )
        claims.append(
            ClaimStatus(
                "A proxy-only calibrated clipping repair recovers quality at fixed candidate budget.",
                "supported" if repair_gain > 0.15 else "weak",
                f"At max N, repair true-score gain is {repair_gain:.3f} with energy tradeoff {repair_energy:.3f}.",
            )
        )

    return claims

