# Final Audit

Audit date: 2026-06-13.

## Thesis

In a controlled Transformer-structured energy landscape, minimum-energy
candidate selection can turn larger candidate budgets into worse semantic
validity: increasing N finds lower proxy-energy candidates, but those candidates
can be locally compatible attention-shortcut artifacts that violate the global
task rule.

## Novelty

The generic claim that sample-and-rank inference can overoptimize a proxy is not
novel. The surviving contribution is a focused, reproducible diagnostic for
Transformer-structured energy models:

- a prompt-conditioned global-validity task;
- a local attention-weighted proxy energy with an explicit shortcut component;
- N-sweep diagnostics for energy-tail exploitation, invalid low-energy states,
  artifact collapse, and proxy/true mismatch;
- a proxy-only calibrated clipping repair that uses the energy decomposition but
  not labels.

## Literature Coverage

`docs/related_work_matrix.csv` contains generated arXiv/OpenReview-backed rows
from `experiments/build_literature_matrix.py`. Coverage includes sample-and-rank
inference, reward hacking, reward-model overoptimization, test-time compute,
verifier search, self-consistency, preference optimization, rejection sampling,
energy text models, energy diffusion language models, EBM sampling, and
Transformer/energy hybrids.

Primary hostile sources considered include:

- EBTs: https://openreview.net/forum?id=ZBj3Qp1bYg
- Inference-time reward hacking: https://arxiv.org/abs/2506.19248
- Reward model overoptimization: https://arxiv.org/abs/2210.10760
- Proximity-regularized sample selection: https://arxiv.org/html/2404.01054v3
- Residual EBMs for text: https://arxiv.org/abs/2004.11714
- EDLM: https://arxiv.org/abs/2410.21357
- Verifier search: https://arxiv.org/abs/2110.14168
- Test-time compute scaling: https://arxiv.org/abs/2408.03314

## Proof Status

The paper includes a simple diagnostic proposition for a mixture model: if a
shortcut candidate class has nonzero proposal probability and uniformly lower
proxy energy than valid candidates, minimum-energy selection chooses a shortcut
with probability `1 - (1 - p)^N`. This is a proof for the idealized construction
only. There is no theorem for arbitrary EBTs.

## Strongest Results

Full experiment: 480 prompt replicates, nested budgets
`N in {1,2,4,8,16,32,64,128}`.

- Minimum-energy selected proxy energy: -0.035 at N=1 to -2.009 at N=128.
- Minimum-energy selected true score: 0.550 at N=1 to 0.170 at N=128.
- Minimum-energy selected validity: 0.429 at N=1 to 0.000 at N=128.
- Minimum-energy artifact selection: 0.156 at N=1 to 1.000 at N=128.
- Calibrated clipping at N=128: true score 0.983, validity 0.981, artifact
  selection 0.019, selected proxy energy -0.065.

`results/claim_status.json` marks the three main claims as supported in the
controlled setup.

## Verification

Commands run successfully:

- `python -m pytest`: 9 passed.
- `python experiments/run_energy_tail_audit.py --preset smoke`: regenerated quick
  CSV/JSON and raw CSV.
- `python experiments/run_energy_tail_audit.py --preset full`: regenerated the
  full summary CSV/JSON, raw CSV, claim-status JSON, and 4 figures used by the
  paper.
- `python experiments/build_literature_matrix.py`: wrote 351-row related work
  matrix.
- `powershell -ExecutionPolicy Bypass -File .\build_paper.ps1` from `paper/`:
  compiled the paper with the official ICLR 2026 LaTeX template and copied the
  final PDF.

The paper now uses `paper/iclr2026_conference.sty` and
`paper/iclr2026_conference.bst` from the official ICLR 2026 template zip. It is
in anonymous submission mode because `paper/energy_tail_audit.tex` does
not invoke `\iclrfinalcopy`, so the rendered author block is "Anonymous authors
/ Paper under double-blind review".

Anonymity checks on the compiled PDF text found no occurrences of
`Jason-Wang313`, `wangz`, `github.com/Jason-Wang313`, or `C:\Users`. PDF
metadata reports blank Author/Title fields.

Final PDF checks:

- Rendered PDF has 5 pages.
- `pdftotext` found no author/path/repo leaks.
- The only remaining `best-of-n` string is the literal title of a cited prior
  work in the references.

Remaining LaTeX warnings after the final rebuild:

- One underfull `\vbox` while outputting page 1.
- MiKTeX emitted routine "have not checked for updates" notices.

## Weaknesses

- Synthetic landscape only.
- The artifact mechanism is intentionally built into the proxy energy.
- No real EBT checkpoint, benchmark task, or human-preference evaluation.
- No comparison to proximity-regularized selectors, hedging methods,
  uncertainty ensembles, or learned verifiers.
- The repair is diagnostic and tuned to the visible shortcut decomposition.

## Delivery

Final PDF:

`C:\Users\wangz\OneDrive\Desktop\best-of-n-ebm-transformer-v2.pdf`

GitHub repository:

https://github.com/Jason-Wang313/best-of-n-ebm-transformer
