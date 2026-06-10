# Best-of-N EBM Transformer

This repository is a CPU-first research artifact for studying Best-of-N inference
in Transformer-structured energy models. It builds a controlled synthetic
landscape where a Transformer-like energy scorer assigns low energy to local
attention-mediated shortcuts, while the true task requires global consistency.

The artifact contains:

- a deterministic toy EBM Transformer landscape in `src/best_of_n_ebm/`;
- Best-of-N and calibrated repair selectors;
- diagnostics for tail exploitation, mode collapse, invalid low-energy states,
  and score/execution mismatch;
- reproducible smoke experiments that generate CSV/JSON summaries and figures;
- an anonymous ICLR-style paper under `paper/`.

## Quickstart

```powershell
python -m pytest
python experiments/run_synthetic_bon.py --preset smoke
python experiments/build_literature_matrix.py
```

Build the paper:

```powershell
cd paper
pdflatex best_of_n_ebm_transformer.tex
bibtex best_of_n_ebm_transformer
pdflatex best_of_n_ebm_transformer.tex
pdflatex best_of_n_ebm_transformer.tex
```

The final delivery PDF is copied to:

```text
C:\Users\wangz\Downloads\best-of-n-ebm-transformer.pdf
```

