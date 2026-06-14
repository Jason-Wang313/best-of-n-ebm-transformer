# Final Audit

Audit date: 2026-06-14.

## Thesis

In a controlled Transformer-structured energy landscape, minimum-energy
candidate selection can turn larger candidate budgets into worse semantic
validity. Increasing N finds lower proxy-energy candidates, but those
candidates can be locally compatible attention-shortcut artifacts that violate
the global prompt-conditioned task rule.

## V3 Contribution

The v3 paper is no longer a short template note. It is a 25-page ICLR-style
submission artifact centered on shortcut-tail audits for Transformer EBMs. The
surviving contribution is a focused diagnostic, not a universal theorem about
all EBTs:

- a prompt-conditioned global-validity task;
- a local attention-weighted proxy energy with an explicit shortcut component;
- high-budget stress to N=512;
- shortcut-prior and global-penalty stress;
- minimum-energy, diversity-constrained, and calibrated clipping selectors;
- repair-grid sensitivity;
- candidate-level energy/true-score calibration;
- failure-case logging and a machine-readable claim audit.

## Strongest V3 Results

The expansion suite in `results/expansion/` reports:

- At N=512, minimum-energy selection reaches selected energy -2.059, selected
  validity 0.000, and artifact selection 1.000.
- At N=512, calibrated clipping reaches selected true score 1.000, selected
  validity 1.000, and artifact selection 0.000 on the same diagnostic family.
- Even with shortcut artifact prior 0.04, minimum-energy selection at N=512
  selects artifacts in every replicate.
- Increasing the global penalty through the tested range improves partial true
  score but leaves selected validity at 0.000 for minimum-energy selection.
- Candidate-level calibration has positive bulk Spearman correlation between
  negative energy and true score, but the lower energy tail is still unsafe.

## Verification

Commands run for the v3 artifact:

- `python -m pytest -q`
- `python -m compileall src tests experiments -q`
- `python experiments\run_expansion_suite.py --mode full --output results\expansion`
- `powershell -ExecutionPolicy Bypass -File paper\build_paper.ps1`
- `python scripts\run_claim_audit.py`

The claim audit reports `submission-ready v3` and checks both the repository
PDF and Desktop PDF.

Final PDF checks:

- Repository PDF: `paper/final/best-of-n-ebm-transformer-v3.pdf`
- Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\best-of-n-ebm-transformer-v3.pdf`
- Page count: 25 pages.
- LaTeX log scan found no undefined references, citation failures, overfull
  boxes, fatal errors, emergency stops, LaTeX warnings, natbib warnings, or
  hyperref warnings.
- Stale-name scan found no v2 Desktop path or unrelated paper names after this
  audit update.

## Weaknesses Kept Explicit

- Synthetic landscape only.
- The artifact mechanism is intentionally built into the proxy energy.
- No real EBT checkpoint, benchmark task, or human-preference evaluation.
- The repair is diagnostic and depends on an observable shortcut feature.
- A trained-model extension should add checkpoints, hidden-task evaluation,
  quantile calibration plots, and stronger selector baselines.

## Delivery

Final PDF:

`C:\Users\wangz\OneDrive\Desktop\best-of-n-ebm-transformer-v3.pdf`

GitHub repository:

https://github.com/Jason-Wang313/best-of-n-ebm-transformer
