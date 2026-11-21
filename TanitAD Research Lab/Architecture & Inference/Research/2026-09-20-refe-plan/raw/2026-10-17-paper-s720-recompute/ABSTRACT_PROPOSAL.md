# Abstract proposal for §7.20 (for the Master Mind; the abstract itself is not edited)

The abstract currently ends its results narrative at the second-round update (v0.3, §7.2). Proposed addition,
three sentences, to follow the latest results paragraph and be tagged as the v1.4 update:

> *Reference-arm update (v1.4, §7.20):* REFe, a reproduction attempt of DriveZero's camera planner, scores
> **88.25 PDMS [87.05, 89.36]** on the full NAVSIM v1.1 navtest (T1 non-reactive pseudo-simulation, 12,146 tokens,
> 136 logs, one training seed) against a published matched row of 94.55 — the bar is **not** cleared. **65.9 %
> of the 6.30-point gap is a tail of 529 zero-scored tokens**, and that tail is a *selection* failure rather than
> a proposal failure (a non-zero proposal existed on 86.0 % [80.1, 91.7] of them), made mostly of sub-half-metre
> near-twin boundary grazes that the scorer's drivable-area head cannot rank (within-token AUC 0.46 against 0.92
> on random tokens). An execution-side lever aimed at the tail was closed by our own measurement,
> which also forced the correction of a gross "reach" figure (1.07 PDMS) that had been quoted as net; the next
> lever, a trunk-level drivable-area gradient (P6), has not yet trained.

Notes for the editor:
* Every number above is in §7.20 with its artifact and estimator; the 88.25 interval is recomputed for this
  section (`SOURCES.md` §0) and not yet in the registry.
* If three sentences are too many, the minimum honest form is the first sentence alone plus "the bar is not
  cleared"; do not keep the PDMS without the bar.
* The abstract's existing claims about the 4B architecture are untouched; REFe is a reference arm (§3.8), not
  that architecture, and the proposed text says so by name.
