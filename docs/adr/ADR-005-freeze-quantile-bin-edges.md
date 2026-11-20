# ADR-005: Freeze quantile bin edges for PSI (the binning war story)

Status: accepted

## Context

PSI compares a production histogram to a reference histogram, which means it needs bin
edges, and the choice of edges is arbitrary in a way that is easy to miss. During the
build a benign variance shift on one feature, a shift that causes no harm, was scored
with two textbook binnings. Under equal-width bins, which split the reference range
into equal intervals, it read 0.22 under equal-width and crossed the 0.2 alarm level.
Under quantile bins, which give each reference bin equal mass, the same shift read
0.14 under quantile and stayed quiet. The alarm fired or not depending on nothing but
the binning strategy, and the scenario it fired on was harmless.

The cause is the tails. A variance increase pushes mass into the extreme intervals,
which equal-width binning fixes on the reference range and which are sparse, so a few
points there move PSI a lot. Quantile bins are balanced by construction and are far
less sensitive to a tail that fattens.

## Decision

Freeze quantile bin edges on the reference window as the default. Keep equal-width
implemented so the sensitivity stays measurable rather than asserted, and pin the flip
with a regression test so a later edit that reintroduces equal-width, or that tunes the
shift until the flip disappears, turns the build red.

## Alternatives considered

- **Equal-width bins.** Rejected: hostage to the reference tails, as the war story
  shows, and the first thing to produce a false alarm on a harmless shift.
- **More bins.** Rejected: raising the bin count sharpens the same sensitivity rather
  than removing it, and trades one arbitrary choice for another.

## Consequences

The default is stable against a fattening tail, and the repository can show, with a
committed number, that an alarm it did not take was an artifact of a choice nobody
should have to make by feel.

### What this cannot measure

Freezing quantile edges removes this particular artifact; it does not make PSI's
threshold principled. The 0.2 alarm level remains a rule of thumb, and this ADR does
not claim an optimal threshold, only a binning that does not flip the verdict on its
own.
