from __future__ import annotations

import json
from pathlib import Path

from experiments.run_digits_benchmark import run


def test_digits_benchmark_quick_writes_passing_claims(tmp_path: Path) -> None:
    manifest = run(quick=True, output_dir=tmp_path / "digits")

    for key in ("trials", "summary", "claims", "figure", "manifest"):
        assert Path(manifest[key]).exists()

    claims = json.loads(Path(manifest["claims"]).read_text(encoding="utf-8"))
    assert claims["all_passed"]
    assert "digits_true_score_drops" in claims["checks"]
    assert "digits_random_baseline_beats_tail" in claims["checks"]
    assert "digits_oracle_pool_contains_valid_alternatives" in claims["checks"]
