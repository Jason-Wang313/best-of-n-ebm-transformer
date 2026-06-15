"""Run the scikit-learn Digits shortcut-tail benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from energy_tail_audit.digits_benchmark import (  # noqa: E402
    FULL_N_VALUES,
    QUICK_N_VALUES,
    make_figure,
    run_digits_benchmark,
    summarize,
    write_outputs,
)


def run(quick: bool = False, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path.cwd()
    artifact_dir = Path(output_dir) if output_dir is not None else root / "results" / "digits_benchmark"
    figure_path = (
        artifact_dir / "figures" / "figure10_digits_benchmark.png"
        if output_dir is not None
        else root / "figures" / "figure10_digits_benchmark.png"
    )
    target_count = 24 if quick else 120
    n_values = QUICK_N_VALUES if quick else FULL_N_VALUES
    rows, meta = run_digits_benchmark(target_count=target_count, n_values=n_values, seed=2026)
    summary = summarize(rows)
    make_figure(summary, figure_path)
    return write_outputs(rows, meta, artifact_dir, figure_path, quick=quick)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Digits shortcut-tail benchmark.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = run(quick=args.quick, output_dir=args.output)
    claims = json.loads(Path(payload["claims"]).read_text(encoding="utf-8"))
    print(f"Digits benchmark: {claims['summary']}")
    print(f"all_passed={claims['all_passed']}")
    print(f"Manifest: {payload['manifest']}")
    return 0 if claims["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
