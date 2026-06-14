from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy_tail_audit.claims import evaluate_claims
from energy_tail_audit.diagnostics import aggregate_rows, pool_diagnostics
from energy_tail_audit.landscape import ToyEBMTransformer
from energy_tail_audit.selection import (
    select_min_energy,
    select_calibrated_clipped,
    select_diversity_constrained,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
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


def preset_config(preset: str) -> tuple[int, int]:
    if preset == "smoke":
        return 24, 4
    if preset == "full":
        return 80, 6
    raise ValueError(f"unknown preset: {preset}")


def run(preset: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seeds, trials_per_seed = preset_config(preset)
    landscape = ToyEBMTransformer()
    methods = {
        "min_energy": select_min_energy,
        "calibrated_clipped": select_calibrated_clipped,
        "diversity_constrained": select_diversity_constrained,
    }

    raw_rows: list[dict[str, object]] = []
    for seed in range(seeds):
        for trial in range(trials_per_seed):
            rng = np.random.default_rng(10_000 * seed + trial)
            prompt = landscape.sample_prompt(rng)
            pool = landscape.sample_candidates(prompt, max(N_VALUES), rng)
            for n in N_VALUES:
                batch = pool.take(n)
                for method_name, selector in methods.items():
                    selected = selector(batch)
                    diag = pool_diagnostics(batch, selected)
                    raw_rows.append(
                        {
                            "preset": preset,
                            "seed": seed,
                            "trial": trial,
                            "N": n,
                            "method": method_name,
                            "selection_index": selected.index,
                            "evaluation_budget": n,
                            **diag,
                        }
                    )

    grouped: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["method"]), int(row["N"]))].append(row)  # type: ignore[arg-type]

    summary_rows: list[dict[str, object]] = []
    for (method, n), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        agg = aggregate_rows(rows, METRIC_KEYS)  # type: ignore[arg-type]
        summary_rows.append(
            {
                "preset": preset,
                "method": method,
                "N": n,
                "replicates": len(rows),
                "candidate_budget": n,
                "evaluation_budget": n,
                **agg,
            }
        )
    return raw_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_metric(
    summary: list[dict[str, object]],
    metric: str,
    ylabel: str,
    output: Path,
    methods: list[str] | None = None,
) -> None:
    methods = methods or sorted({str(row["method"]) for row in summary})
    plt.figure(figsize=(6.0, 3.8))
    for method in methods:
        rows = sorted(
            [row for row in summary if row["method"] == method],
            key=lambda row: int(row["N"]),
        )
        x = np.array([int(row["N"]) for row in rows])
        y = np.array([float(row[f"{metric}_mean"]) for row in rows])
        err = np.array([float(row[f"{metric}_stderr"]) for row in rows])
        plt.plot(x, y, marker="o", label=method.replace("_", " "))
        plt.fill_between(x, y - 1.96 * err, y + 1.96 * err, alpha=0.14)
    plt.xscale("log", base=2)
    plt.xlabel("candidate budget N")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    raw_rows, summary_rows = run(args.preset)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    write_csv(RESULTS / f"{args.preset}_raw.csv", raw_rows)
    write_csv(RESULTS / f"{args.preset}_summary.csv", summary_rows)
    write_json(
        RESULTS / f"{args.preset}_summary.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "preset": args.preset,
            "n_values": N_VALUES,
            "summary": summary_rows,
        },
    )

    claims = evaluate_claims(summary_rows)
    write_json(
        RESULTS / "claim_status.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "claims": [claim.__dict__ for claim in claims],
        },
    )

    plot_metric(
        summary_rows,
        "selected_true_score",
        "selected true score",
        FIGURES / "validity_vs_n.png",
        ["min_energy", "calibrated_clipped", "diversity_constrained"],
    )
    plot_metric(
        summary_rows,
        "selected_energy",
        "selected proxy energy",
        FIGURES / "energy_tail_vs_n.png",
        ["min_energy", "calibrated_clipped"],
    )
    plot_metric(
        summary_rows,
        "artifact_selected",
        "artifact selection rate",
        FIGURES / "artifact_selection_vs_n.png",
        ["min_energy", "calibrated_clipped", "diversity_constrained"],
    )
    plot_metric(
        summary_rows,
        "score_execution_spearman",
        "pool proxy/true rank correlation",
        FIGURES / "score_mismatch_vs_n.png",
        ["min_energy"],
    )

    print(f"Wrote {RESULTS / f'{args.preset}_summary.csv'}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
