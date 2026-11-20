# ADR-001: A synthetic seeded generator, not a real production log

Status: accepted

## Context

The question the project answers is whether a population-drift alarm predicts model
harm. To answer it you must, for each production window, know the true cause of any
difference from reference: did the inputs move, did the labelling rule move, both, or
neither. A real production log cannot tell you this. You observe inputs, and
eventually labels, but never the counterfactual, so you can never separate a benign
covariate shift from a concept shift that happened to coincide with one. On real data
the central claim is not just hard to measure, it is unfalsifiable.

## Decision

Generate the world. A seeded generator draws inputs from a fixed reference
distribution and labels from a fixed concept, and every scenario is an explicit,
named departure from that world: a covariate shift moves P(X), a concept shift moves
P(Y|X), a mixed shift moves both, a null moves neither. The deployed model is fit
once on the reference world and frozen. The suite is 43 labelled scenarios across the
four families, with the concept family carrying the real degradations.

## Alternatives considered

- **A public tabular dataset with an injected shift.** Rejected: injecting a shift
  into real data still leaves you unable to prove the injected cause is the only
  cause, and a reviewer cannot re-derive the ground truth.
- **A replayed real log.** Rejected for the unfalsifiability above, and because it
  would need network or a large committed warehouse, breaking the offline contract.

## Consequences

Every number is reproducible from one command with no network, and a reviewer can
read the generator and see exactly what each scenario did. The cost is the boundary
this ADR exists to state.

### What this cannot measure

A synthetic world cannot tell you the *prevalence* of concept drift versus covariate
drift in any real system, so this repository never claims one. It establishes a
structural fact (a label-blind detector cannot see a label-only change) and measures
its size under known conditions; it does not estimate how often that condition arises
in production. The generator's inputs are independent Gaussians and the concept is
logistic, which is simpler than most real data; a richer model class would change the
exact AUC drops but not the structural result, because the result follows from the
detector's inputs, not from the shape of the data.
