from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_tail_audit.diagnostics import aggregate_rows, pool_diagnostics, spearman_like
from energy_tail_audit.landscape import CandidateBatch, ToyEBMTransformer
from energy_tail_audit.selection import (
    SelectionResult,
    select_calibrated_clipped,
    select_diversity_constrained,
    select_min_energy,
)


RESULTS = ROOT / "results" / "expansion"
FIGURES = ROOT / "figures"

METRIC_KEYS = [
    "selected_energy",
    "selected_true_score",
    "selected_valid",
    "selected_shortcut",
    "energy_tail_gap",
    "invalid_low_energy_rate",
    "pool_valid_rate",
    "pool_true_score",
    "score_execution_spearman",
    "mode_entropy",
    "selected_mode_frequency",
    "artifact_selected",
]


@dataclass(frozen=True)
class SuiteConfig:
    mode: str
    seeds: int
    trials_per_seed: int
    n_values: tuple[int, ...]
    stress_seeds: int
    stress_trials: int


PRESETS = {
    "quick": SuiteConfig(
        mode="quick-smoke",
        seeds=4,
        trials_per_seed=2,
        n_values=(1, 8, 32),
        stress_seeds=2,
        stress_trials=1,
    ),
    "full": SuiteConfig(
        mode="full-v3",
        seeds=32,
        trials_per_seed=4,
        n_values=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        stress_seeds=12,
        stress_trials=3,
    ),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summary(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)  # type: ignore[arg-type]

    out: list[dict[str, object]] = []
    for key, bucket in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        agg = aggregate_rows(bucket, METRIC_KEYS)  # type: ignore[arg-type]
        base = {name: value for name, value in zip(group_keys, key)}
        base["replicates"] = len(bucket)
        out.append({**base, **agg})
    return out


def _record_selection(
    *,
    regime: str,
    condition: str,
    seed: int,
    trial: int,
    n_value: int,
    method: str,
    batch: CandidateBatch,
    selected: SelectionResult,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "regime": regime,
        "condition": condition,
        "seed": seed,
        "trial": trial,
        "N": n_value,
        "method": method,
        "selection_index": selected.index,
        **pool_diagnostics(batch, selected),
    }
    if extra:
        row.update(extra)
    return row


def _method_selectors() -> dict[str, Callable[[CandidateBatch], SelectionResult]]:
    return {
        "min_energy": select_min_energy,
        "calibrated_clipped": select_calibrated_clipped,
        "diversity_constrained": select_diversity_constrained,
    }


def _candidate_rows(batch: CandidateBatch, *, seed: int, trial: int, regime: str, condition: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(batch.tokens.shape[0]):
        rows.append(
            {
                "regime": regime,
                "condition": condition,
                "seed": seed,
                "trial": trial,
                "candidate_index": idx,
                "proposal_type": str(batch.proposal_type[idx]),
                "proxy_energy": float(batch.proxy_energy[idx]),
                "true_score": float(batch.true_score[idx]),
                "valid": int(batch.valid[idx]),
                "violations": int(batch.violations[idx]),
                "attention_shortcut": float(batch.attention_shortcut[idx]),
                "attention_entropy": float(batch.attention_entropy[idx]),
                "repetition": float(batch.repetition[idx]),
                "local_attention_energy": float(batch.local_attention_energy[idx]),
                "global_penalty": float(batch.global_penalty[idx]),
            }
        )
    return rows


def _run_budget_sweep(config: SuiteConfig) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = ToyEBMTransformer()
    rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    max_n = max(config.n_values)
    selectors = _method_selectors()
    for seed in range(config.seeds):
        for trial in range(config.trials_per_seed):
            rng = np.random.default_rng(90_000 + seed * 100 + trial)
            prompt = model.sample_prompt(rng)
            pool = model.sample_candidates(prompt, max_n, rng)
            candidates.extend(_candidate_rows(pool, seed=seed, trial=trial, regime="budget_sweep", condition="base"))
            for n_value in config.n_values:
                batch = pool.take(n_value)
                for method, selector in selectors.items():
                    selected = selector(batch)
                    rows.append(
                        _record_selection(
                            regime="budget_sweep",
                            condition="base",
                            seed=seed,
                            trial=trial,
                            n_value=n_value,
                            method=method,
                            batch=batch,
                            selected=selected,
                        )
                    )
    return rows, candidates


def _run_prior_sweep(config: SuiteConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_n = max(config.n_values)
    for artifact_prob in [0.04, 0.14, 0.28, 0.42]:
        near_valid = max(0.30, 0.88 - artifact_prob)
        model = ToyEBMTransformer()
        condition = f"artifact_{artifact_prob:.2f}"
        for seed in range(config.stress_seeds):
            for trial in range(config.stress_trials):
                rng = np.random.default_rng(120_000 + seed * 100 + trial)
                prompt = model.sample_prompt(rng)
                pool = model.sample_candidates(
                    prompt,
                    max_n,
                    rng,
                    near_valid_prob=near_valid,
                    artifact_prob=artifact_prob,
                )
                batch = pool.take(max_n)
                for method, selector in _method_selectors().items():
                    selected = selector(batch)
                    rows.append(
                        _record_selection(
                            regime="artifact_prior",
                            condition=condition,
                            seed=seed,
                            trial=trial,
                            n_value=max_n,
                            method=method,
                            batch=batch,
                            selected=selected,
                            extra={"artifact_prob": artifact_prob, "near_valid_prob": near_valid},
                        )
                    )
    return rows


def _run_global_weight_sweep(config: SuiteConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    max_n = max(config.n_values)
    for global_weight in [0.25, 0.62, 1.00, 1.40]:
        model = ToyEBMTransformer()
        model.global_weight = global_weight
        condition = f"global_weight_{global_weight:.2f}"
        for seed in range(config.stress_seeds):
            for trial in range(config.stress_trials):
                rng = np.random.default_rng(150_000 + seed * 100 + trial)
                prompt = model.sample_prompt(rng)
                pool = model.sample_candidates(prompt, max_n, rng)
                batch = pool.take(max_n)
                for method, selector in _method_selectors().items():
                    selected = selector(batch)
                    rows.append(
                        _record_selection(
                            regime="global_weight",
                            condition=condition,
                            seed=seed,
                            trial=trial,
                            n_value=max_n,
                            method=method,
                            batch=batch,
                            selected=selected,
                            extra={"global_weight": global_weight},
                        )
                    )
    return rows


def _run_repair_grid(config: SuiteConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    model = ToyEBMTransformer()
    max_n = max(config.n_values)
    for seed in range(config.stress_seeds):
        for trial in range(config.stress_trials):
            rng = np.random.default_rng(180_000 + seed * 100 + trial)
            prompt = model.sample_prompt(rng)
            batch = model.sample_candidates(prompt, max_n, rng).take(max_n)
            for clip_quantile in [0.05, 0.20, 0.40]:
                for shortcut_weight in [0.0, 0.5, 1.0, 2.0]:
                    selected = select_calibrated_clipped(
                        batch,
                        clip_quantile=clip_quantile,
                        shortcut_weight=shortcut_weight,
                    )
                    rows.append(
                        _record_selection(
                            regime="repair_grid",
                            condition=f"q{clip_quantile:.2f}_w{shortcut_weight:.1f}",
                            seed=seed,
                            trial=trial,
                            n_value=max_n,
                            method="calibrated_clipped",
                            batch=batch,
                            selected=selected,
                            extra={"clip_quantile": clip_quantile, "shortcut_weight": shortcut_weight},
                        )
                    )
    return rows


def _failure_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    budget_rows = [row for row in rows if row["regime"] == "budget_sweep"]
    max_n = max(int(row["N"]) for row in budget_rows)
    high = [
        row
        for row in rows
        if row["regime"] == "budget_sweep" and int(row["N"]) == max_n and row["method"] in {"min_energy", "calibrated_clipped"}
    ]
    high.sort(key=lambda row: (float(row["selected_true_score"]), -float(row["selected_shortcut"])))
    return high[:40]


def _candidate_correlation(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_type[str(row["proposal_type"])].append(row)
        by_type["all"].append(row)
    for proposal_type, bucket in sorted(by_type.items()):
        energy = np.array([float(row["proxy_energy"]) for row in bucket])
        true_score = np.array([float(row["true_score"]) for row in bucket])
        valid = np.array([float(row["valid"]) for row in bucket])
        rows.append(
            {
                "proposal_type": proposal_type,
                "count": len(bucket),
                "spearman_neg_energy_true": spearman_like(-energy, true_score),
                "energy_mean": float(energy.mean()),
                "true_score_mean": float(true_score.mean()),
                "valid_rate": float(valid.mean()),
            }
        )
    return rows


def _plot_budget(summary: list[dict[str, object]], output: Path) -> None:
    rows = [row for row in summary if row["regime"] == "budget_sweep" and row["condition"] == "base"]
    methods = ["min_energy", "calibrated_clipped", "diversity_constrained"]
    colors = {"min_energy": "#b23a48", "calibrated_clipped": "#2f7f6f", "diversity_constrained": "#667085"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7), constrained_layout=True)
    for method in methods:
        group = sorted([row for row in rows if row["method"] == method], key=lambda row: int(row["N"]))
        x = np.array([int(row["N"]) for row in group])
        axes[0].plot(x, [float(row["selected_energy_mean"]) for row in group], marker="o", color=colors[method], label=method)
        axes[1].plot(x, [float(row["selected_true_score_mean"]) for row in group], marker="o", color=colors[method])
        axes[2].plot(x, [float(row["artifact_selected_mean"]) for row in group], marker="o", color=colors[method])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate budget N")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("selected proxy energy")
    axes[1].set_ylabel("selected true score")
    axes[2].set_ylabel("artifact selection rate")
    axes[0].set_title("Lower energy tail")
    axes[1].set_title("Semantic validity")
    axes[2].set_title("Shortcut collapse")
    axes[0].legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_condition(summary: list[dict[str, object]], regime: str, x_key: str, output: Path, title: str) -> None:
    rows = [row for row in summary if row["regime"] == regime]
    methods = ["min_energy", "calibrated_clipped", "diversity_constrained"]
    colors = {"min_energy": "#b23a48", "calibrated_clipped": "#2f7f6f", "diversity_constrained": "#667085"}
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), constrained_layout=True)
    for method in methods:
        group = sorted([row for row in rows if row["method"] == method], key=lambda row: float(row[x_key]))
        if not group:
            continue
        x = np.array([float(row[x_key]) for row in group])
        axes[0].plot(x, [float(row["selected_true_score_mean"]) for row in group], marker="o", color=colors[method], label=method)
        axes[1].plot(x, [float(row["artifact_selected_mean"]) for row in group], marker="o", color=colors[method], label=method)
    axes[0].set_ylabel("selected true score")
    axes[1].set_ylabel("artifact selection rate")
    for ax in axes:
        ax.set_xlabel(x_key.replace("_", " "))
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_repair_grid(summary: list[dict[str, object]], output: Path) -> None:
    rows = [row for row in summary if row["regime"] == "repair_grid"]
    qs = sorted({float(row["clip_quantile"]) for row in rows})
    ws = sorted({float(row["shortcut_weight"]) for row in rows})
    matrix = np.zeros((len(qs), len(ws)))
    for row in rows:
        i = qs.index(float(row["clip_quantile"]))
        j = ws.index(float(row["shortcut_weight"]))
        matrix[i, j] = float(row["selected_true_score_mean"])
    fig, ax = plt.subplots(figsize=(5.2, 3.9), constrained_layout=True)
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(ws)), [str(w) for w in ws])
    ax.set_yticks(range(len(qs)), [str(q) for q in qs])
    ax.set_xlabel("shortcut weight")
    ax.set_ylabel("clip quantile")
    ax.set_title("Repair grid true score")
    fig.colorbar(im, ax=ax, shrink=0.82)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _plot_calibration(candidates: list[dict[str, object]], output: Path) -> None:
    data = candidates
    if len(data) > 7000:
        rng = np.random.default_rng(1729)
        idx = rng.choice(len(data), size=7000, replace=False)
        data = [data[int(i)] for i in idx]
    colors = {"valid": "#2f7f6f", "near_valid": "#3f5f9f", "artifact": "#b23a48", "random": "#667085"}
    fig, ax = plt.subplots(figsize=(5.4, 4.1), constrained_layout=True)
    for proposal_type in sorted({str(row["proposal_type"]) for row in data}):
        group = [row for row in data if row["proposal_type"] == proposal_type]
        ax.scatter(
            [float(row["proxy_energy"]) for row in group],
            [float(row["true_score"]) for row in group],
            s=7,
            alpha=0.18,
            color=colors.get(proposal_type, "#333333"),
            label=proposal_type,
        )
    ax.set_xlabel("proxy energy")
    ax.set_ylabel("true score")
    ax.set_title("Low energy is not always high truth")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _claims(summary: list[dict[str, object]], correlations: list[dict[str, object]], mode: str) -> dict[str, object]:
    def get(regime: str, method: str, n_value: int, metric: str, condition: str = "base") -> float:
        rows = [
            row
            for row in summary
            if row["regime"] == regime
            and row["condition"] == condition
            and row["method"] == method
            and int(row["N"]) == n_value
        ]
        if not rows:
            return float("nan")
        return float(rows[0][metric])

    max_n = 32 if mode == "quick-smoke" else 512
    min_e_1 = get("budget_sweep", "min_energy", 1, "selected_energy_mean")
    min_e_max = get("budget_sweep", "min_energy", max_n, "selected_energy_mean")
    min_valid_1 = get("budget_sweep", "min_energy", 1, "selected_valid_mean")
    min_valid_max = get("budget_sweep", "min_energy", max_n, "selected_valid_mean")
    min_artifact_max = get("budget_sweep", "min_energy", max_n, "artifact_selected_mean")
    repair_true_max = get("budget_sweep", "calibrated_clipped", max_n, "selected_true_score_mean")
    min_true_max = get("budget_sweep", "min_energy", max_n, "selected_true_score_mean")
    repair_artifact_max = get("budget_sweep", "calibrated_clipped", max_n, "artifact_selected_mean")
    all_corr = next(row for row in correlations if row["proposal_type"] == "all")

    checks = {
        "min_energy_enters_lower_tail": min_e_max <= min_e_1 - (0.40 if mode == "full-v3" else 0.05),
        "min_energy_validity_collapses": min_valid_max <= min_valid_1 - (0.25 if mode == "full-v3" else 0.01),
        "min_energy_artifact_rate_high": min_artifact_max >= (0.90 if mode == "full-v3" else 0.20),
        "repair_improves_true_score": repair_true_max >= min_true_max + (0.30 if mode == "full-v3" else 0.05),
        "repair_reduces_artifacts": repair_artifact_max <= min_artifact_max - (0.50 if mode == "full-v3" else 0.05),
        "proxy_truth_rank_mismatch_exists": float(all_corr["spearman_neg_energy_true"]) < 0.65,
    }
    return {
        "mode": mode,
        "claim_pass": bool(all(checks.values())),
        "checks": checks,
        "key_numbers": {
            "min_energy_n1_energy": min_e_1,
            "min_energy_nmax_energy": min_e_max,
            "min_energy_n1_validity": min_valid_1,
            "min_energy_nmax_validity": min_valid_max,
            "min_energy_nmax_artifact_rate": min_artifact_max,
            "min_energy_nmax_true_score": min_true_max,
            "repair_nmax_true_score": repair_true_max,
            "repair_nmax_artifact_rate": repair_artifact_max,
            "candidate_spearman_neg_energy_true": float(all_corr["spearman_neg_energy_true"]),
        },
    }


def run_suite(config: SuiteConfig, output: Path = RESULTS) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rows, candidates = _run_budget_sweep(config)
    rows.extend(_run_prior_sweep(config))
    rows.extend(_run_global_weight_sweep(config))
    rows.extend(_run_repair_grid(config))

    summary = _summary(rows, ("regime", "condition", "method", "N"))
    # Add stress metadata to summary rows for plotting.
    metadata: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        metadata.setdefault((str(row["regime"]), str(row["condition"])), {}).update(
            {k: v for k, v in row.items() if k in {"artifact_prob", "near_valid_prob", "global_weight", "clip_quantile", "shortcut_weight"}}
        )
    for row in summary:
        row.update(metadata.get((str(row["regime"]), str(row["condition"])), {}))

    correlations = _candidate_correlation(candidates)
    failures = _failure_cases(rows)
    claims = _claims(summary, correlations, config.mode)

    trials_path = output / "expanded_trials.csv"
    summary_path = output / "expanded_summary.csv"
    candidates_path = output / "candidate_diagnostics.csv"
    correlation_path = output / "correlation_summary.csv"
    failures_path = output / "failure_cases.csv"
    claims_path = output / "claims.json"
    manifest_path = output / "manifest.json"

    _write_csv(trials_path, rows)
    _write_csv(summary_path, summary)
    _write_csv(candidates_path, candidates)
    _write_csv(correlation_path, correlations)
    _write_csv(failures_path, failures)
    _write_json(claims_path, claims)
    _write_json(
        manifest_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": config.mode,
            "n_values": list(config.n_values),
            "seeds": config.seeds,
            "trials_per_seed": config.trials_per_seed,
            "stress_seeds": config.stress_seeds,
            "stress_trials": config.stress_trials,
        },
    )

    _plot_budget(summary, FIGURES / "figure5_budget_512.png")
    _plot_condition(summary, "artifact_prior", "artifact_prob", FIGURES / "figure6_artifact_prior.png", "Shortcut prior stress")
    _plot_condition(summary, "global_weight", "global_weight", FIGURES / "figure7_global_weight.png", "Global penalty stress")
    _plot_repair_grid(summary, FIGURES / "figure8_repair_grid.png")
    _plot_calibration(candidates, FIGURES / "figure9_energy_calibration.png")

    for figure in [
        "figure5_budget_512.png",
        "figure6_artifact_prior.png",
        "figure7_global_weight.png",
        "figure8_repair_grid.png",
        "figure9_energy_calibration.png",
    ]:
        (output / figure).write_bytes((FIGURES / figure).read_bytes())

    return {
        "trials": trials_path,
        "summary": summary_path,
        "candidates": candidates_path,
        "correlation": correlation_path,
        "failures": failures_path,
        "claims": claims_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3 shortcut-tail expansion diagnostics.")
    parser.add_argument("--mode", choices=sorted(PRESETS), default="full")
    parser.add_argument("--output", type=Path, default=RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_suite(PRESETS[args.mode], args.output)
    claims = json.loads(paths["claims"].read_text(encoding="utf-8"))
    print(f"expansion suite complete: {claims['mode']}; claim_pass={claims['claim_pass']}")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
