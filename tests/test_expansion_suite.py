from __future__ import annotations

import json

import numpy as np

from experiments.run_expansion_suite import PRESETS, run_suite
from energy_tail_audit.landscape import ToyEBMTransformer
from energy_tail_audit.selection import select_calibrated_clipped, select_min_energy


def test_high_budget_min_energy_finds_shortcut_tail() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(2026)
    prompt = model.sample_prompt(rng)
    pool = model.sample_candidates(prompt, 256, rng)
    low = select_min_energy(pool.take(8))
    high = select_min_energy(pool.take(256))
    assert high.proxy_energy <= low.proxy_energy
    assert high.attention_shortcut >= low.attention_shortcut


def test_calibrated_repair_accepts_parameter_grid() -> None:
    model = ToyEBMTransformer()
    rng = np.random.default_rng(7)
    prompt = model.sample_prompt(rng)
    batch = model.sample_candidates(prompt, 128, rng)
    selected = select_calibrated_clipped(batch, clip_quantile=0.40, shortcut_weight=2.0)
    assert 0 <= selected.index < 128


def test_quick_expansion_suite_writes_claims(tmp_path) -> None:
    paths = run_suite(PRESETS["quick"], tmp_path)
    for path in paths.values():
        assert path.exists()
    claims = json.loads(paths["claims"].read_text(encoding="utf-8"))
    assert claims["mode"] == "quick-smoke"
    assert "min_energy_enters_lower_tail" in claims["checks"]
