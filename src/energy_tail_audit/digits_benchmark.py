"""CPU-light Digits benchmark for shortcut-tail EBM selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits


FULL_N_VALUES = (1, 4, 16, 64, 128)
QUICK_N_VALUES = (1, 4, 16, 64)


def load_digit_targets(target_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a deterministic subset of the standard scikit-learn Digits data."""

    x, y = load_digits(return_X_y=True)
    images = (x.reshape(-1, 8, 8) / 16.0).astype(float)
    labels = y.astype(int)
    prototypes = np.stack([images[labels == digit].mean(axis=0) for digit in range(10)])
    eligible = [
        idx
        for idx, image in enumerate(images)
        if image[4:, :].mean() > 0.12 and image[4:, :].std() > 0.16
    ]
    if target_count > len(eligible):
        raise ValueError(f"target_count={target_count} exceeds eligible digits={len(eligible)}")
    selected = np.asarray(eligible, dtype=int)[
        np.linspace(0, len(eligible) - 1, int(target_count), dtype=int)
    ]
    return images, labels, prototypes, selected


def run_digits_benchmark(
    *,
    target_count: int = 120,
    n_values: Sequence[int] = FULL_N_VALUES,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images, labels, prototypes, selected = load_digit_targets(target_count)
    rows: list[dict[str, Any]] = []
    max_n = int(max(n_values))
    for target_id, source_idx in enumerate(selected):
        rng = np.random.default_rng(seed + int(source_idx) * 17)
        target = images[int(source_idx)]
        label = int(labels[int(source_idx)])
        pool = _sample_pool(
            target=target,
            label=label,
            images=images,
            labels=labels,
            prototypes=prototypes,
            max_n=max_n,
            rng=rng,
        )
        for n_value in n_values:
            candidates = pool[: int(n_value)]
            for method in ("min_energy", "calibrated_clipped"):
                selected_idx, score_used = _select(candidates, method)
                row = {
                    **candidates[selected_idx],
                    "benchmark": "sklearn_digits_hidden_completion",
                    "target_id": int(target_id),
                    "source_idx": int(source_idx),
                    "digit": label,
                    "N": int(n_value),
                    "method": method,
                    "selection_index": int(selected_idx),
                    "score_used": float(score_used),
                }
                rows.append(row)
    meta = {
        "benchmark": "sklearn_digits_hidden_completion",
        "dataset": "scikit-learn load_digits / UCI Optical Recognition of Handwritten Digits",
        "target_count": int(target_count),
        "candidate_counts": [int(value) for value in n_values],
        "seed": int(seed),
        "observed_region": "top four rows of each 8x8 digit",
        "held_out_region": "bottom four rows of each 8x8 digit",
    }
    return rows, meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        for n_value in sorted({int(row["N"]) for row in rows}):
            group = [row for row in rows if row["method"] == method and int(row["N"]) == n_value]
            if not group:
                continue
            item: dict[str, Any] = {"method": method, "N": int(n_value), "count": len(group)}
            for key in (
                "proxy_energy",
                "true_score",
                "valid",
                "shortcut_mass",
                "hidden_mse",
                "prototype_error",
                "smoothness",
                "seam_error",
            ):
                item[f"{key}_mean"] = float(np.mean([float(row[key]) for row in group]))
            item["shortcut_rate"] = float(np.mean([row["proposal_type"] == "shortcut" for row in group]))
            out.append(item)
    return out


def claim_gates(summary: list[dict[str, Any]], *, quick: bool = False) -> dict[str, Any]:
    def row(method: str, n_value: int) -> dict[str, Any]:
        matches = [item for item in summary if item["method"] == method and int(item["N"]) == n_value]
        if not matches:
            raise KeyError((method, n_value))
        return matches[0]

    max_n = max(int(item["N"]) for item in summary)
    min_1 = row("min_energy", 1)
    min_high = row("min_energy", max_n)
    repair_high = row("calibrated_clipped", max_n)
    thresholds = {
        "energy_drop": -0.30 if not quick else -0.15,
        "true_drop": -0.20 if not quick else -0.15,
        "validity": 0.10 if not quick else 0.20,
        "shortcut_rate": 0.90 if not quick else 0.80,
        "repair_true_gain": 0.25,
        "repair_shortcut_reduction": 0.70,
    }
    checks = {
        "digits_energy_tail_decreases": _claim(
            min_high["proxy_energy_mean"] - min_1["proxy_energy_mean"],
            thresholds["energy_drop"],
            "<",
            "Minimum-energy selection enters a lower proxy-energy tail on real digit completions.",
        ),
        "digits_true_score_drops": _claim(
            min_high["true_score_mean"] - min_1["true_score_mean"],
            thresholds["true_drop"],
            "<",
            "The lower energy tail has worse held-out bottom-half reconstruction score.",
        ),
        "digits_validity_collapses": _claim(
            min_high["valid_mean"],
            thresholds["validity"],
            "<",
            "High-budget minimum-energy selections rarely satisfy the held-out target threshold.",
        ),
        "digits_shortcut_rate_high": _claim(
            min_high["shortcut_rate"],
            thresholds["shortcut_rate"],
            ">",
            "High-budget minimum-energy selection is dominated by shortcut completions.",
        ),
        "digits_repair_recovers_true_score": _claim(
            repair_high["true_score_mean"] - min_high["true_score_mean"],
            thresholds["repair_true_gain"],
            ">",
            "Calibrated clipping recovers held-out completion score without using held-out pixels.",
        ),
        "digits_repair_reduces_shortcuts": _claim(
            min_high["shortcut_rate"] - repair_high["shortcut_rate"],
            thresholds["repair_shortcut_reduction"],
            ">",
            "Calibrated clipping reduces shortcut-completion selection at high budget.",
        ),
    }
    return {
        "all_passed": all(payload["passed"] for payload in checks.values()),
        "checks": checks,
        "summary": (
            f"energy change {min_high['proxy_energy_mean'] - min_1['proxy_energy_mean']:.3f}, "
            f"true-score change {min_high['true_score_mean'] - min_1['true_score_mean']:.3f}, "
            f"repair gain {repair_high['true_score_mean'] - min_high['true_score_mean']:.3f}."
        ),
    }


def make_figure(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)
    colors = {"min_energy": "#b23a48", "calibrated_clipped": "#2f7f6f"}
    labels = {"min_energy": "minimum energy", "calibrated_clipped": "calibrated clipping"}
    for method in ("min_energy", "calibrated_clipped"):
        group = sorted([row for row in summary if row["method"] == method], key=lambda item: int(item["N"]))
        x = np.array([int(row["N"]) for row in group])
        axes[0].plot(x, [row["proxy_energy_mean"] for row in group], marker="o", color=colors[method], label=labels[method])
        axes[1].plot(x, [row["true_score_mean"] for row in group], marker="o", color=colors[method])
        axes[2].plot(x, [row["shortcut_rate"] for row in group], marker="o", color=colors[method])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("candidate budget N")
    axes[0].set_title("selected proxy energy")
    axes[1].set_title("held-out true score")
    axes[2].set_title("shortcut selection")
    axes[0].set_ylabel("mean value")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Digits hidden-completion shortcut-tail benchmark")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_outputs(
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
    output_dir: Path,
    figure_path: Path,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    claims = claim_gates(summary, quick=quick)
    trials_path = output_dir / "digits_trials.csv"
    summary_path = output_dir / "digits_summary.csv"
    claims_path = output_dir / "claims.json"
    manifest_path = output_dir / "manifest.json"
    _write_csv(trials_path, rows)
    _write_csv(summary_path, summary)
    _write_json(claims_path, claims)
    manifest = {
        **meta,
        "quick": bool(quick),
        "trials": str(trials_path),
        "summary": str(summary_path),
        "claims": str(claims_path),
        "figure": str(figure_path),
        "all_passed": claims["all_passed"],
    }
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def _sample_pool(
    *,
    target: np.ndarray,
    label: int,
    images: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    same_label = np.flatnonzero(labels == label)
    for candidate_idx in range(max_n):
        draw = rng.random()
        full = target.copy()
        if draw < 0.58:
            proposal_type = "near_valid"
            hidden = target[4:, :].copy() + rng.normal(0.0, 0.08, size=(4, 8))
            mask = rng.random((4, 8)) < 0.08
            if mask.any():
                hidden[mask] = np.clip(
                    hidden[mask] + rng.normal(0.0, 0.35, size=int(mask.sum())),
                    0.0,
                    1.0,
                )
            full[4:, :] = np.clip(hidden, 0.0, 1.0)
        elif draw < 0.73:
            proposal_type = "class_neighbor"
            source = int(rng.choice(same_label))
            hidden = images[source, 4:, :] + rng.normal(0.0, 0.04, size=(4, 8))
            full[4:, :] = np.clip(hidden, 0.0, 1.0)
        elif draw < 0.87:
            proposal_type = "shortcut"
            full[4:, :] = _shortcut_hidden(rng)
        else:
            proposal_type = "random_digit"
            source = int(rng.integers(0, len(images)))
            full[4:, :] = images[source, 4:, :]

        features = _features(full, label, prototypes)
        hidden_mse = float(np.mean((full[4:, :] - target[4:, :]) ** 2))
        true_score = float(max(0.0, 1.0 - hidden_mse / 0.35))
        pool.append(
            {
                "candidate_index": int(candidate_idx),
                "proposal_type": proposal_type,
                "proxy_energy": features["proxy_energy"],
                "true_score": true_score,
                "valid": float(true_score >= 0.80),
                "shortcut_mass": features["shortcut_mass"],
                "hidden_mse": hidden_mse,
                "prototype_error": features["prototype_error"],
                "smoothness": features["smoothness"],
                "seam_error": features["seam_error"],
            }
        )
    return pool


def _features(full: np.ndarray, label: int, prototypes: np.ndarray) -> dict[str, float]:
    hidden = full[4:, :]
    horizontal = np.abs(hidden[:, 1:] - hidden[:, :-1]).mean()
    vertical = np.abs(hidden[1:, :] - hidden[:-1, :]).mean()
    smoothness = float((horizontal + vertical) / 2.0)
    seam_error = float(np.abs(full[3, :] - full[4, :]).mean())
    prototype_error = float(np.mean((hidden - prototypes[label, 4:, :]) ** 2))
    row_repetition = 1.0 - float(vertical)
    col_repetition = 1.0 - float(horizontal)
    low_variance = max(0.0, 1.0 - float(hidden.std()) * 4.0)
    shortcut_mass = 0.45 * row_repetition + 0.25 * col_repetition + 0.30 * low_variance
    proxy_energy = (
        0.55 * smoothness
        + 0.20 * seam_error
        + 0.20 * prototype_error
        - 0.75 * shortcut_mass
    )
    return {
        "proxy_energy": float(proxy_energy),
        "shortcut_mass": float(shortcut_mass),
        "prototype_error": prototype_error,
        "smoothness": smoothness,
        "seam_error": seam_error,
    }


def _shortcut_hidden(rng: np.random.Generator) -> np.ndarray:
    mode = str(rng.choice(["blank", "stripe", "block"], p=[0.45, 0.35, 0.20]))
    hidden = np.zeros((4, 8), dtype=float)
    if mode == "stripe":
        hidden[:] = float(rng.uniform(0.0, 0.18))
        col = int(rng.integers(1, 7))
        hidden[:, col : col + 1] = float(rng.uniform(0.45, 0.85))
    elif mode == "block":
        hidden[:] = float(rng.uniform(0.0, 0.10))
        hidden[:, 2:6] = float(rng.uniform(0.35, 0.75))
    else:
        hidden[:] = float(rng.uniform(0.0, 0.12))
    return np.clip(hidden, 0.0, 1.0)


def _select(candidates: list[dict[str, Any]], method: str) -> tuple[int, float]:
    energy = np.asarray([float(row["proxy_energy"]) for row in candidates])
    shortcut = np.asarray([float(row["shortcut_mass"]) for row in candidates])
    if method == "min_energy":
        idx = int(np.argmin(energy))
        return idx, float(energy[idx])
    if method != "calibrated_clipped":
        raise ValueError(f"unknown selector: {method}")
    lower = float(np.quantile(energy, 0.20))
    shortcut_center = float(np.median(shortcut))
    score = np.maximum(energy, lower) + 0.90 * np.maximum(0.0, shortcut - shortcut_center)
    idx = int(np.argmin(score))
    return idx, float(score[idx])


def _claim(value: float, threshold: float, op: str, description: str) -> dict[str, Any]:
    passed = value > threshold if op == ">" else value < threshold
    return {
        "passed": bool(passed),
        "observed": float(value),
        "threshold": float(threshold),
        "op": op,
        "description": description,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
