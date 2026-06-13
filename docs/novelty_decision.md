# Novelty Decision

Decision date: 2026-06-10.

## Initial Thesis

Minimum-energy candidate selection in Transformer-structured energy models can amplify
low-energy artifacts: as N grows, selected candidates increasingly occupy
pathological low-energy regions that are not semantically valid or robust under
perturbation. The candidate mechanism is attention-mediated compositional energy
shortcuts, where local token compatibility dominates global consistency.

## Literature Verdict

The generic version of this thesis is not novel. Reward hacking, reward-model
overoptimization, verifier search, and proximity-regularized selection are all
active and well-covered areas.

The surviving novelty is narrower:

1. Diagnose a Transformer-structured EBM failure mode, not a generic reward
   model failure.
2. Use a controlled synthetic landscape where the true task requires global
   prompt-conditioned consistency but the proxy energy is built from local
   attention-weighted compatibility terms.
3. Track N-dependent movement into the low-energy tail with diagnostics for
   invalid low-energy states, attention-shortcut mass, mode collapse, and
   proxy/true mismatch.
4. Evaluate a proxy-only repair that uses the energy decomposition, not true
   labels: calibrated lower-tail clipping plus attention-shortcut penalty.

## Final Thesis For The Paper

In a controlled Transformer-structured energy landscape, minimum-energy
selection can reliably turn more candidate budget into worse semantic validity:
as N increases from 1 to 128, the selected energy becomes much lower while the
selected candidate collapses into locally compatible, globally invalid attention
shortcut motifs. A simple proxy-only calibrated clipping selector prevents this
failure in the toy setting at the same candidate/evaluation budget, but with a
large energy-optimality tradeoff.

## What We Must Not Claim

- We do not claim deployed EBTs fail.
- We do not claim inference-time reward hacking is new.
- We do not claim the repair is generally better than proximity penalties,
  uncertainty hedging, or verifier calibration.
- We do not claim benchmark-scale evidence.
- We do not claim a theorem beyond the controlled construction and diagnostics.

## Strongest Supported Result

Full experiment, 480 prompt replicates:

- Minimum-energy selected proxy energy moves from -0.035 at N=1 to -2.009 at
  N=128.
- Minimum-energy selected true score moves from 0.550 to 0.170.
- Minimum-energy selected validity moves from 0.429 to 0.000.
- Minimum-energy artifact selection moves from 0.156 to 1.000.
- Calibrated clipping at N=128 reaches true score 0.983 and validity 0.981,
  with selected proxy energy -0.065, so the repair trades away extreme proxy
  optimality.
