from __future__ import annotations

import inspect

import numpy as np

from best_of_n_ebm.claims import evaluate_claims
from best_of_n_ebm.diagnostics import pool_diagnostics
from best_of_n_ebm.landscape import ToyEBMTransformer
from best_of_n_ebm.selection import select_best_of_n, select_calibrated_clipped


def test_best_of_n_selects_minimum_proxy_energy() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(11)
    prompt = model.sample_prompt(rng)
    batch = model.sample_candidates(prompt, 32, rng)
    selected = select_best_of_n(batch)
    assert selected.index == int(np.argmin(batch.proxy_energy))


def test_nested_best_of_n_energy_is_nonincreasing() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(12)
    prompt = model.sample_prompt(rng)
    pool = model.sample_candidates(prompt, 128, rng)
    energies = [select_best_of_n(pool.take(n)).proxy_energy for n in [1, 2, 4, 8, 16, 32, 64, 128]]
    assert all(next_energy <= energy for energy, next_energy in zip(energies, energies[1:]))


def test_calibrated_clipping_does_not_read_labels() -> None:
    source = inspect.getsource(select_calibrated_clipped)
    assert ".valid" not in source
    assert ".true_score" not in source
    assert ".violations" not in source


def test_repair_uses_same_candidate_budget() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(13)
    prompt = model.sample_prompt(rng)
    batch = model.sample_candidates(prompt, 64, rng)
    bon = select_best_of_n(batch)
    repaired = select_calibrated_clipped(batch)
    assert 0 <= bon.index < 64
    assert 0 <= repaired.index < 64


def test_diagnostic_rows_support_expected_failure_on_toy_setting() -> None:
    model = ToyEBMTransformer()
    rows = []
    for seed in range(16):
        rng = np.random.default_rng(seed)
        prompt = model.sample_prompt(rng)
        pool = model.sample_candidates(prompt, 128, rng)
        for n in [1, 128]:
            batch = pool.take(n)
            selected = select_best_of_n(batch)
            rows.append({"N": n, "method": "bon", **pool_diagnostics(batch, selected)})
    low_n = [row for row in rows if row["N"] == 1]
    high_n = [row for row in rows if row["N"] == 128]
    assert np.mean([row["selected_energy"] for row in high_n]) < np.mean(
        [row["selected_energy"] for row in low_n]
    )
    assert np.mean([row["selected_true_score"] for row in high_n]) < np.mean(
        [row["selected_true_score"] for row in low_n]
    )


def test_claim_evaluator_marks_strong_synthetic_rows_supported() -> None:
    rows = [
        {
            "method": "bon",
            "N": 1,
            "selected_energy_mean": 0.1,
            "selected_valid_mean": 0.9,
            "selected_true_score_mean": 0.9,
        },
        {
            "method": "bon",
            "N": 128,
            "selected_energy_mean": -0.8,
            "selected_valid_mean": 0.3,
            "selected_true_score_mean": 0.4,
        },
        {
            "method": "calibrated_clipped",
            "N": 1,
            "selected_energy_mean": 0.1,
            "selected_valid_mean": 0.9,
            "selected_true_score_mean": 0.9,
        },
        {
            "method": "calibrated_clipped",
            "N": 128,
            "selected_energy_mean": -0.2,
            "selected_valid_mean": 0.7,
            "selected_true_score_mean": 0.72,
        },
    ]
    claims = evaluate_claims(rows)
    assert any(claim.status == "supported" for claim in claims)

