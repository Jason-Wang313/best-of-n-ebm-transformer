from __future__ import annotations

import inspect

import numpy as np

from energy_tail_audit.claims import evaluate_claims
from energy_tail_audit.diagnostics import pool_diagnostics
from energy_tail_audit.landscape import ToyEBMTransformer
from energy_tail_audit.selection import select_min_energy, select_calibrated_clipped


def test_min_energy_selects_minimum_proxy_energy() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(11)
    prompt = model.sample_prompt(rng)
    batch = model.sample_candidates(prompt, 32, rng)
    selected = select_min_energy(batch)
    assert selected.index == int(np.argmin(batch.proxy_energy))


def test_nested_min_energy_is_nonincreasing() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(12)
    prompt = model.sample_prompt(rng)
    pool = model.sample_candidates(prompt, 128, rng)
    energies = [select_min_energy(pool.take(n)).proxy_energy for n in [1, 2, 4, 8, 16, 32, 64, 128]]
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
    min_energy = select_min_energy(batch)
    repaired = select_calibrated_clipped(batch)
    assert 0 <= min_energy.index < 64
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
            selected = select_min_energy(batch)
            rows.append({"N": n, "method": "min_energy", **pool_diagnostics(batch, selected)})
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
            "method": "min_energy",
            "N": 1,
            "selected_energy_mean": 0.1,
            "selected_valid_mean": 0.9,
            "selected_true_score_mean": 0.9,
        },
        {
            "method": "min_energy",
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
