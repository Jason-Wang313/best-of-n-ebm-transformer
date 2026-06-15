# Reviewer Attacks

## Attack: The task is synthetic, so the result is not evidence about EBTs.

Concede partially. The replicated mechanism benchmark is synthetic, but v4 adds
a scikit-learn Digits hidden-completion tier with held-out real pixels. The
paper is still a diagnostic, not a deployed-model benchmark. The intended value
is to specify a minimal failure test that real EBT implementations should pass.

## Attack: The model bakes in the artifact.

Yes, by design. The point is not to discover artifacts by accident; it is to
make the shortcut mechanism observable and falsifiable. The paper should say
"can" and "in this construction", not "will."

## Attack: The repair has access to privileged information.

The repair only uses proxy-side quantities: candidate energies and the
attention-shortcut component of the energy decomposition. Tests assert that the
selector does not read `valid`, `true_score`, or `violations`.

## Attack: The repair just clips away all low-energy candidates.

Mostly true in this toy setting, and that is why the paper reports the energy
tradeoff. The repair is best read as a diagnostic intervention: if clipping the
tail restores validity, the low-energy tail was suspect.

## Attack: The paper ignores stronger baselines like proximity-regularized selection and hedging.

The related work names them as stronger general-purpose mitigation lines. This
artifact's baseline is intentionally local: minimum-energy selection versus a
mechanism-aware proxy-only selector on the same candidates. A benchmark paper
would need those baselines.

## Attack: Results are too small for ICLR.

The v4 paper expands the evidence substantially: 25+ pages, N=512 stress,
shortcut-prior sweeps, global-penalty sweeps, repair-grid sensitivity,
candidate-level calibration, failure-case logging, and a real-data Digits tier.
The remaining fair attack is the absence of trained EBT checkpoints, not the
absence of a substantial diagnostic artifact.

## Attack: The paper is a duplicate candidate-pool wrapper.

The response is to keep the paper EBM-specific throughout: energy
decomposition, local attention compatibility, shortcut mass, lower-tail energy
calibration, and repair by energy clipping plus shortcut-feature penalty. The
title, abstract, figures, tables, and appendix all avoid presenting the result
as a generic best-of-N theorem.
