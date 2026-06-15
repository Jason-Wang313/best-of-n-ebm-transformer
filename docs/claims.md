# Claim Boundary

Generated v4 claim status is written to `results/expansion/claims.json`.
The real-data Digits tier writes `results/digits_benchmark/claims.json`.

## Supported By Current Full Results

- Minimum-energy selection drives selected candidates into lower proxy-energy tails.
- In the controlled landscape, lower proxy energy correlates with lower semantic
  validity at high N.
- The selected high-N minimum-energy candidates are dominated by artifact proposals.
- A proxy-only calibrated clipping repair can recover validity in this toy
  setting at the same candidate budget.
- The N=512 expansion, shortcut-prior stress, global-penalty stress, repair
  grid, and candidate-level calibration all support the narrow diagnostic claim.
- The scikit-learn Digits hidden-completion tier shows the same low-energy-tail
  failure on recognized real inputs with held-out bottom-half pixels.

## Weak Or Out Of Scope

- Generalization to real EBT checkpoints.
- Trained-checkpoint comparison to proximity-regularized selectors, hedging
  methods, process verifiers, or uncertainty ensembles.
- Claims about human preference, safety, or actual language quality.
- Formal proof for arbitrary Transformer energies.

## Claim-To-Artifact Map

- Energy tail movement: `figures/energy_tail_vs_n.png`,
  `results/full_summary.csv`, `results/expansion/expanded_summary.csv`.
- Validity collapse: `figures/validity_vs_n.png`,
  `results/full_summary.csv`, `figures/figure5_budget_512.png`.
- Artifact selection: `figures/artifact_selection_vs_n.png`.
- Shortcut-prior stress: `figures/figure6_artifact_prior.png`.
- Global-penalty stress: `figures/figure7_global_weight.png`.
- Repair sensitivity: `figures/figure8_repair_grid.png`.
- Proxy/true mismatch: `figures/score_mismatch_vs_n.png`,
  `figures/figure9_energy_calibration.png`.
- Repair status: `results/claim_status.json`,
  `results/expansion/claims.json`.
- Real-data tier: `figures/figure10_digits_benchmark.png`,
  `results/digits_benchmark/digits_summary.csv`,
  `results/digits_benchmark/claims.json`.
- Non-label-leaking selector test:
  `tests/test_selection_and_claims.py::test_calibrated_clipping_does_not_read_labels`.
