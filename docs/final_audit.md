# Final Audit

Audit date: 2026-06-10.

## Thesis

In a controlled Transformer-structured energy landscape, vanilla Best-of-N
selection can turn larger candidate budgets into worse semantic validity:
increasing N finds lower proxy-energy candidates, but those candidates can be
locally compatible attention-shortcut artifacts that violate the global task
rule.

## Novelty

The generic claim that BoN can overoptimize a proxy is not novel. The surviving
contribution is a focused, reproducible diagnostic for Transformer-structured
energy models:

- a prompt-conditioned global-validity task;
- a local attention-weighted proxy energy with an explicit shortcut component;
- N-sweep diagnostics for energy-tail exploitation, invalid low-energy states,
  artifact collapse, and proxy/true mismatch;
- a proxy-only calibrated clipping repair that uses the energy decomposition but
  not labels.

## Literature Coverage

`docs/related_work_matrix.csv` contains 351 rows generated from live arXiv
queries by `experiments/build_literature_matrix.py`. Coverage includes BoN,
reward hacking, reward-model overoptimization, test-time compute, verifier
search, self-consistency, preference optimization, rejection sampling, energy
text models, energy diffusion language models, EBM sampling, and
Transformer/energy hybrids.

Primary hostile sources considered include:

- EBTs: https://openreview.net/forum?id=ZBj3Qp1bYg
- Inference-time reward hacking: https://arxiv.org/abs/2506.19248
- Reward model overoptimization: https://arxiv.org/abs/2210.10760
- RBoN: https://arxiv.org/html/2404.01054v3
- Residual EBMs for text: https://arxiv.org/abs/2004.11714
- EDLM: https://arxiv.org/abs/2410.21357
- Verifier search: https://arxiv.org/abs/2110.14168
- Test-time compute scaling: https://arxiv.org/abs/2408.03314

## Proof Status

The paper includes a simple diagnostic proposition for a mixture model: if a
shortcut candidate class has nonzero proposal probability and uniformly lower
proxy energy than valid candidates, BoN selects a shortcut with probability
`1 - (1 - p)^N`. This is a proof for the idealized construction only. There is
no theorem for arbitrary EBTs.

## Strongest Results

Smoke experiment: 96 prompt replicates, nested budgets
`N in {1,2,4,8,16,32,64,128}`.

- Vanilla BoN selected proxy energy: -0.078 at N=1 to -1.999 at N=128.
- Vanilla BoN selected true score: 0.578 at N=1 to 0.155 at N=128.
- Vanilla BoN selected validity: 0.458 at N=1 to 0.000 at N=128.
- Vanilla BoN artifact selection: 0.167 at N=1 to 1.000 at N=128.
- Calibrated clipping at N=128: true score 1.000, validity 1.000, artifact
  selection 0.000, selected proxy energy -0.032.

`results/claim_status.json` marks the three main claims as supported in the
controlled setup.

## Verification

Commands run successfully:

- `python -m pytest`: 9 passed.
- `python experiments/run_synthetic_bon.py --preset smoke`: regenerated summary
  CSV/JSON, raw CSV, claim-status JSON, and 4 figures.
- `python experiments/build_literature_matrix.py`: wrote 351-row related work
  matrix.
- `powershell -ExecutionPolicy Bypass -File .\build_paper.ps1` from `paper/`:
  compiled the paper and copied the final PDF.

MiKTeX emitted a routine update notice and a small overfull hbox warning, but
the PDF compiled successfully.

## Weaknesses

- Synthetic landscape only.
- The artifact mechanism is intentionally built into the proxy energy.
- No real EBT checkpoint, benchmark task, or human-preference evaluation.
- No comparison to RBoN, HedgeTune, uncertainty ensembles, or learned verifiers.
- The repair is diagnostic and tuned to the visible shortcut decomposition.

## Delivery

Final PDF:

`C:\Users\wangz\Downloads\best-of-n-ebm-transformer.pdf`

GitHub repository:

https://github.com/Jason-Wang313/best-of-n-ebm-transformer

