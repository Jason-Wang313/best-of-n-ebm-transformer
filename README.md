# Shortcut-Energy Audits for Transformer EBMs

This repository is a CPU-first research artifact for auditing shortcut-energy
tails in Transformer-structured energy models. It builds a controlled synthetic
landscape where a Transformer-like energy scorer assigns very low energy to
local attention-mediated shortcuts, while the true task requires global
prompt-conditioned consistency.

The v3 artifact contains:

- a deterministic toy EBM Transformer landscape in `src/energy_tail_audit/`;
- minimum-energy, calibrated clipping, and diversity-constrained selectors;
- diagnostics for tail exploitation, mode collapse, invalid low-energy states,
  and score/execution mismatch;
- a high-budget expansion suite with N=512 stress, shortcut-prior sweeps,
  global-penalty sweeps, repair-grid sensitivity, candidate-level calibration,
  and failure-case logging;
- reproducible CSV/JSON summaries, figures, and a machine-readable claim
  certificate;
- a 25-page anonymous ICLR-style v3 paper under `paper/`.

## Quickstart

```powershell
python -m pytest -q
python experiments/run_energy_tail_audit.py --preset smoke
python experiments/run_expansion_suite.py --mode full --output results\expansion
python experiments/build_literature_matrix.py
```

Build the paper:

```powershell
powershell -ExecutionPolicy Bypass -File paper\build_paper.ps1
python scripts\run_claim_audit.py
```

The final delivery PDF is copied to:

```text
C:\Users\wangz\OneDrive\Desktop\best-of-n-ebm-transformer-v3.pdf
```
