# ADR-004: Score the drift alarm as a classifier of harm, and price every zero

Status: accepted

## Context

A monitoring policy that alarms on drift is implicitly predicting harm, so the fair
way to judge it is as a classifier: over a scenario set, how often does an alarm
coincide with real harm (precision), and how often does harm occur with no alarm
(recall's complement). Reporting only the cases where drift and harm agree would
flatter the detector; reporting a bare recall of zero would overclaim in the other
direction, because a measured zero over a finite sample is not a true zero.

## Decision

Build the 2x2 of alarm against harm over the measurable scenarios, and report both
precision at the suite's real base rate and recall, restricted to each family. Price
every zero against its resolution floor: a recall of zero over k harmed scenarios is
reported as "no higher than one in k could show", with k stated. Scenarios whose harm
could not be measured (a degenerate one-class window) are counted apart and kept out
of the rates, so "could not answer" never moves precision.

## Decision in numbers

Across 23 harmed scenarios, the generous drift alarm reaches a precision of 50% as a
harm predictor, at a harm prevalence of 53% of the suite. A precision at or below the
base rate means the alarm carries no information about harm: being told it fired does
not raise your estimate that harm occurred. Of the 23 harmed scenarios, 16 of 23 are
silent to PSI, which is nearly 70% of realised harm invisible to the detector until
labels arrive.

## Alternatives considered

- **Report accuracy of the alarm.** Rejected: accuracy is dominated by the easy true
  negatives and hides the two numbers that matter, the misses and the false alarms.
- **Report recall as a bare zero.** Rejected: without the resolution floor and the
  sample size, a zero approves every downstream claim, which is the most common way a
  portfolio overclaims.

## Consequences

The confusion framing turns a flattering "the alarm agrees with harm half the time"
into the honest "the alarm is no better than a coin weighted to the base rate", and
makes the silent-failure share the headline.

### What this cannot measure

Precision and recall here are conditional on this suite's family mix, which is a
design choice, not an estimate of production. The numbers say what the alarm is worth
*on these scenarios*; they do not forecast its precision on a real stream, whose mix
of covariate and concept drift is unknown and unmeasurable from inputs alone.
