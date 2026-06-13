# Literature Map

Sweep date: 2026-06-10.

This map is grounded in the generated matrix at `docs/related_work_matrix.csv`,
which currently contains 351 arXiv/OpenReview-backed rows. The matrix was built
with `python experiments/build_literature_matrix.py` and covers these themes:
sample-and-rank inference, reward hacking, reward-model overoptimization, test-time compute,
verifier search, self-consistency, preference optimization, rejection sampling,
energy-based text models, energy-based diffusion language models,
Transformer/energy hybrids, and EBM sampling.

## Core EBT Prior

Energy-Based Transformers (EBTs) were introduced as models that learn to score
input/candidate compatibility and make predictions by energy minimization. The
OpenReview record lists the paper as an ICLR 2026 Oral, published January 26,
2026, and positions EBTs as a general System 2 inference-time compute mechanism
across discrete and continuous modalities:

- OpenReview: https://openreview.net/forum?id=ZBj3Qp1bYg
- arXiv: https://arxiv.org/abs/2507.02092

Implication for this project: broad claims that "energy minimization can improve
inference" are not novel. The usable opening is failure analysis: what happens
when a simple sample-and-select policy is bolted onto a Transformer-structured
energy model whose energy is a misspecified proxy?

## Sample-And-Rank Inference-Time Alignment

The sample-and-rank literature is already large. It establishes that selecting
from multiple candidates can be effective, that it can be distilled, and that it
has theoretically analyzable alignment policies:

- Distillation for sample-and-rank policies: https://arxiv.org/abs/2407.14622
- Proximity-regularized selection: https://arxiv.org/html/2404.01054v3
- Inference-time reward hacking evaluation: https://arxiv.org/abs/2502.12668
- Inference-aware fine-tuning: https://arxiv.org/abs/2412.15287

Implication: a paper cannot claim to discover sample-and-rank selection, proxy
overoptimization, or generic regularization. It can claim a controlled
EBT-specific diagnostic if the mechanism is not just "reward model imperfect."

## Reward Hacking And Goodhart Prior

The strongest hostile prior is reward-model overoptimization. Gao et al. study
gold-vs-proxy reward under optimization pressure in a synthetic setting:

- Scaling Laws for Reward Model Overoptimization:
  https://arxiv.org/abs/2210.10760

Khalaf et al. characterize inference-time reward hacking under sample selection
and related mechanisms and introduce hedging procedures:

- Inference-Time Reward Hacking in Large Language Models:
  https://arxiv.org/abs/2506.19248

Implication: the final claim must not be "selection can reward hack." The
contribution must instead identify a mechanism that is natural in
Transformer-structured energies: attention-mediated compositional shortcuts can
create low-energy local regions that are globally invalid.

## Test-Time Compute And Verifiers

Verifier-based selection is a standard route for using inference-time compute.
GSM8K verifier work generates many candidate solutions and selects the highest
ranked completion:

- Training Verifiers to Solve Math Word Problems:
  https://arxiv.org/abs/2110.14168

Snell et al. compare methods for scaling test-time compute, including search
against dense verifier reward models:

- Scaling LLM Test-Time Compute Optimally:
  https://arxiv.org/abs/2408.03314

Self-consistency samples multiple reasoning paths and marginalizes answers:

- Self-Consistency Improves Chain of Thought Reasoning:
  https://arxiv.org/abs/2203.11171

Implication: the artifact should avoid benchmarking claims about test-time
compute. The contribution is diagnostic and mechanistic.

## Energy-Based Text And Diffusion Models

Text EBMs predate EBTs. Residual EBMs for text generation use sequence-level
unnormalized energies and importance sampling:

- Residual Energy-Based Models for Text Generation:
  https://arxiv.org/abs/2004.11714

Energy-Based Diffusion Language Models introduce a sequence-level EBM at each
diffusion step and also use parallel important sampling:

- EDLM: https://arxiv.org/abs/2410.21357

Implication: energy-based reranking and importance sampling for text are also
not novel. This project should focus on the interaction between minimum-energy
selection pressure and Transformer-like local attention energy decomposition.

## Surviving Gap

The viable gap after the sweep is:

> A controlled shortcut-energy audit for Transformer-structured energy models,
> showing how local attention-mediated shortcut energies can dominate global
> semantic validity as N grows, plus a proxy-only calibrated clipping repair
> that improves the toy setting at equal candidate budget.

This is deliberately narrower than the initial ambition. It is a first-pass
mechanism paper, not a benchmark paper and not a proof that deployed EBTs fail.
