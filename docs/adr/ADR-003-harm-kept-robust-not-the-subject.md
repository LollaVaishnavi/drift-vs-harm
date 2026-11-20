# ADR-003: Keep the harm measurement robust, because it is not the subject

Status: accepted

## Context

Harm is the labelled truth the detector is being judged against: a drop in the
model's ranking quality on production data. It would be easy to make harm itself the
hard part, by measuring AUC on a tiny labelled sample where the estimate is so noisy
that a drop cannot be told from sampling variance. That is a real and important
problem, but it is a different one, and a sister project (rag-regression-gate) is
devoted to it. If harm measurement were noisy here, a reader could not tell whether a
missed degradation was the detector's blindness or the metric's noise, and the two
stories would be confounded.

## Decision

Make harm cleanly measurable so the only thing in question is the detector. Measure
AUC on a large labelled window, use a threshold-free primary metric (AUC), corroborate
with log loss and expected calibration error, and set a deliberately modest harm bar
(a 0.02 AUC drop). Across the suite the harm prevalence is 53% of the suite, and the
concept family's real degradations span from 0.08 to a worst silent AUC drop where
AUC falls 0.30.

## Alternatives considered

- **A small labelled sample.** Rejected: it reintroduces metric noise and confounds
  the result with the variance problem another project owns.
- **Accuracy as the primary metric.** Rejected: accuracy moves with the decision
  threshold and the class balance, so a threshold-free ranking metric is the honest
  primary, with calibration as a corroborator.

## Consequences

When the detector misses a concept-drift scenario, the miss cannot be blamed on a
noisy harm number, because the harm number is well resolved. The headline is therefore
about blindness, cleanly.

### What this cannot measure

Harm here requires labels and enough of them. In production those labels arrive late
or never, which is the entire reason drift monitoring exists; this repository measures
what a drift alarm is worth *given* that harm is eventually measurable, and does not
address the separate case where the labelled sample is too small to measure harm at
all.
