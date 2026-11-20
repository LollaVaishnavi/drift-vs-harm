# ADR-002: Watch every feature and the score, and alarm generously

Status: accepted

## Context

A drift monitor has to choose what to watch and how to combine it. The temptation is
to watch only a summary, such as the model's output score, because it is one number.
But the point of the project is to give drift its best possible chance to predict
harm before judging it. The most generous monitor watches every input feature and the
score, runs three detectors (PSI, KS, Jensen-Shannon) on each, and alarms if any of
them fires. If even that monitor misses concept drift, no quieter one can do better.

## Decision

Watch all six features and the model score. Alarm on the union: any detector on any
signal. Report the per-detector aggregates separately, because the detectors fail
differently and averaging them hides it.

## Alternatives considered

- **Score-only monitoring.** Rejected as too weak to be a fair test: it watches one
  signal, so it misses covariate shifts the feature monitor would catch, and it still
  cannot see concept drift.
- **A single detector.** Rejected for the same reason; three detectors on every
  signal is the strongest standard monitor, and the one worth refuting.

## Consequences

The generous union is powerful at firing and useless at discriminating. On pure-null
windows, where nothing changed, PSI alarms on 0% of null windows, but KS, run across
eight signals at the textbook 0.05, alarms on 10% of null windows through multiple
comparisons alone. Bonferroni correction across the signals removes those false
alarms, but, as ADR-004 and the experiments show, it does nothing for the misses: a
tighter threshold cannot give a detector an input it never had. This is why the
headline uses PSI, whose reading on a label-only change is a deterministic zero,
rather than the noisier union.

### What this cannot measure

Watching more signals raises the false-alarm rate but cannot raise recall for concept
drift above zero, so this ADR's generosity buys breadth against covariate shift and
nothing against the silent class.
