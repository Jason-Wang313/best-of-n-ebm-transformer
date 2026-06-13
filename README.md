# Shortcut-Energy Audits for Transformer EBMs

This repository is a CPU-first research artifact for auditing shortcut-energy
tails in Transformer-structured energy models. It builds a controlled synthetic
landscape where a Transformer-like energy scorer assigns very low energy to
local attention-mediated shortcuts, while the true task requires global
prompt-conditioned consistency.

The artifact contains:

- a deterministic toy EBM Transformer landscape in `src/energy_tail_audit/`;
- minimum-energy, calibrated clipping, and diversity-constrained selectors;
- diagnostics for tail exploitation, mode collapse, invalid low-energy states,
  and score/execution mismatch;
- reproducible smoke experiments that generate CSV/JSON summaries and figures;
- an anonymous ICLR-style paper under `paper/`.

## Quickstart

```powershell
python -m pytest
python experiments/run_energy_tail_audit.py --preset smoke
python experiments/build_literature_matrix.py
```

Build the paper:

```powershell
powershell -ExecutionPolicy Bypass -File paper\build_paper.ps1
```

The final delivery PDF is copied to:

```text
C:\Users\wangz\OneDrive\Desktop\best-of-n-ebm-transformer-v2.pdf
```
