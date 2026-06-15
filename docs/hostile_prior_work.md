# Hostile Prior Work

This file records the prior work most likely to make the project look
incremental, and how the final paper narrows itself around that risk.

## 1. "This is just reward hacking."

Strong prior:

- Gao, Schulman, and Hilton, "Scaling Laws for Reward Model Overoptimization",
  arXiv:2210.10760.
- Khalaf et al., "Inference-Time Reward Hacking in Large Language Models",
  arXiv:2506.19248.

Why it is hostile: these papers already show that optimizing an imperfect proxy
with inference-time selection can reduce true reward. Khalaf et al. show a broad
reward-hacking pattern for inference-time mechanisms.

Response: the final contribution is not the generic Goodhart effect. The toy
landscape isolates an EBT-shaped mechanism: a Transformer-like attention energy
decomposition assigns extreme low energies to locally compatible shortcut
motifs, while the true rule depends on global prompt-conditioned consistency.

## 2. "Proximity regularization already fixes this."

Strong prior:

- Jinnai et al., "Regularized sample selection to mitigate reward hacking for
  language model alignment", arXiv:2404.01054.

Why it is hostile: proximity regularization during selection is directly
targeted at inference-time reward hacking.

Response: this artifact's repair is narrower and diagnostic. It does not claim
to dominate proximity-regularized selection. It shows that, when the failure is
localized in an attention shortcut component of an energy model, a proxy-only
tail clipping plus shortcut penalty can recover validity at equal candidate
budget in the controlled setup.

## 3. "EBMs and residual text energies already use importance sampling."

Strong prior:

- Deng et al., "Residual Energy-Based Models for Text Generation",
  arXiv:2004.11714.
- Xu et al., "Energy-Based Diffusion Language Models for Text Generation",
  arXiv:2410.21357.

Why it is hostile: energy-based reranking, residual sequence energies, and
parallel importance sampling are established.

Response: the project is not claiming a new sampling algorithm for EBMs. It
examines a failure mode of selecting the minimum energy from a growing candidate
pool when the energy has Transformer-like local shortcut structure.

## 4. "Verifier search and test-time compute are already well studied."

Strong prior:

- Cobbe et al., "Training Verifiers to Solve Math Word Problems",
  arXiv:2110.14168.
- Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective
  than Scaling Model Parameters", arXiv:2408.03314.
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in
  Language Models", arXiv:2203.11171.

Why it is hostile: sampling multiple candidates and choosing by a verifier is
an old and successful template.

Response: the final paper avoids claiming a better test-time compute method. It
claims an auditing diagnostic for a specific class of energy scorers.

## 5. "EBTs themselves already study thinking longer."

Strong prior:

- Gladstone et al., "Energy-Based Transformers are Scalable Learners and
  Thinkers", OpenReview ICLR 2026 Oral.

Why it is hostile: EBTs already frame inference as energy minimization and
report gains from inference-time computation.

Response: this repo does not evaluate the EBT paper's model. It proposes a small
controlled stress test that should be run before trusting naive minimum-energy
selection over any Transformer-structured energy. The v4 manuscript also adds a
CPU-light Digits hidden-completion benchmark with held-out real pixels, but it
still frames the result as an early warning rather than a refutation of trained
EBT checkpoints.
