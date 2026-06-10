# Reviewer Attacks

## Attack: The task is synthetic, so the result is not evidence about EBTs.

Concede partially. The paper is a mechanism diagnostic, not a deployed-model
benchmark. The intended value is to specify a minimal failure test that real EBT
implementations should pass.

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

## Attack: The paper ignores stronger baselines like RBoN and HedgeTune.

The related work names them as stronger general-purpose mitigation lines. This
artifact's baseline is intentionally local: vanilla BoN versus a mechanism-aware
proxy-only selector on the same candidates. A benchmark paper would need those
baselines.

## Attack: Results are too small for ICLR.

Fair. The repo produces an anonymous ICLR-style paper, but the evidence level is
closer to a workshop mechanism note or negative result. The final audit should
make that explicit.

