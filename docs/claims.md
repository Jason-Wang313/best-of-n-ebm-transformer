# Claim Boundary

Generated claim status is written to `results/claim_status.json`.

## Supported By Current Smoke Results

- Vanilla BoN drives selected candidates into lower proxy-energy tails.
- In the controlled landscape, lower proxy energy correlates with lower semantic
  validity at high N.
- The selected high-N vanilla candidates are dominated by artifact proposals.
- A proxy-only calibrated clipping repair can recover validity in this toy
  setting at the same candidate budget.

## Weak Or Out Of Scope

- Generalization to real EBT checkpoints.
- Comparison to RBoN, HedgeTune, process verifiers, or uncertainty ensembles.
- Claims about human preference, safety, or actual language quality.
- Formal proof for arbitrary Transformer energies.

## Claim-To-Artifact Map

- Energy tail movement: `figures/energy_tail_vs_n.png`,
  `results/smoke_summary.csv`.
- Validity collapse: `figures/validity_vs_n.png`,
  `results/smoke_summary.csv`.
- Artifact selection: `figures/artifact_selection_vs_n.png`.
- Proxy/true mismatch: `figures/score_mismatch_vs_n.png`.
- Repair status: `results/claim_status.json`.
- Non-label-leaking selector test:
  `tests/test_selection_and_claims.py::test_calibrated_clipping_does_not_read_labels`.

