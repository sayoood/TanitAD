# PREREG_V8F_PROOF_ADD133 — GAPSPREADB-T0: the read of card ADD132 with the permuted spread as its own arm

Frozen before any tool of this card is written and before any value of this card or of card ADD132 exists: the tool of card ADD132 has read synthetic test data alone. Tier **T0** (recorded frames, no rollout; a diagnostic of a perception read and not driving performance). CPU only, existing files on the fork pod.

## 1. Why this card, and when it runs

The tool test of card ADD132 showed on synthetic data that the second clause of its control s3 (a permuted spread has to read a ratio near 1 against the flat margin) fails with no mark present whenever the label rows and the other rows differ in their mean spread, or the factors differ much from row to row: the dose is set over all rows with an output, the measure is read on the label rows, and the long part is a convex measure. The clause was written to exclude a false mark and cannot tell it from these two effects.

Order, committed here: card ADD132 is run first, as frozen. If all its controls pass, its name stands and this card is not run. If its control s3 fails, card ADD132 is reported as failed with no name and no value opened, and this card is run; its row is read by the other session before it is landed (fork rule of row 136). If another control of card ADD132 fails (s0, s1, s2 or s4), the same control fails here and both cards are reported as failed.

## 2. Files, rows and arms

As card ADD132, section 2: the base file `VGAPLE.npz` and the three fit files of card ADD106; the label rows of the T0 read of card ADD131 (3,841 rows of 141 clips); m, s, **A90** and **SP90** (factor max(0.60, 1 - k s), k set with no label so that the mean factor over all rows with an output is 0.90). One arm more:

* **PM** — SP90 with the spread permuted, seed 133: among the label rows, and among the other rows with an output. PM has the k of SP90 and on the label rows the same set of factors; the placement alone differs.

## 3. Measures and estimator

Long part L and short part S as in card ADD132. Against the flat margin: RL = L(SP90) / L(A90), RS = S(SP90) / S(A90). Against the permuted arm: RLp = L(SP90) / L(PM), RSp = S(SP90) / S(PM). Each with the cluster bootstrap over the 141 clips, 2,000 draws, seed 133, 95 % interval. Beside, no name: the values listed in card ADD132, the mean factor of SP90 on the label rows, and L(PM) / L(A90), S(PM) / S(A90). Families as in card ADD132: LONGITUDINAL read as the long and the short part of the gap; LATERAL, TACTICAL, STRATEGIC not read (n = 0).

## 4. Names

Against the flat margin, one name: `GAPSPREADB-T0-PLACED-BETTER-THAN-FLAT` (interval of RL below 1 and interval of RS not above 1), `GAPSPREADB-T0-PLACED-TRADES` (interval of RL below 1 and interval of RS above 1), `GAPSPREADB-T0-PLACED-WORSE-THAN-FLAT` (interval of RL above 1), `GAPSPREADB-T0-PLACED-NOT-SEPARATED` (else). With `BETTER-THAN-FLAT`: `GAPSPREADB-T0-BAR-MET` when RL is at most 0.90, else `GAPSPREADB-T0-BAR-NOT-MET`.

On the placement, one name: `GAPSPREADB-T0-SPREAD-MARKS-LONG-READS` (interval of RLp below 1), `GAPSPREADB-T0-SPREAD-MARKS-THE-OTHER-WAY` (interval of RLp above 1), `GAPSPREADB-T0-SPREAD-DOES-NOT-MARK` (else).

## 5. Controls (all have to pass before a name or any value is opened)

* **s0 files**, **s1 rows**, **s2 dose** — as in card ADD132, by the tool of that card, imported unchanged.
* **s3 the measure can read a mark** — with the planted spread of card ADD132, RL lies below 0.90 and the interval of RLp (planted against the planted spread permuted) lies below 1; two permutations of the real spread, seeds 133 and 134, read against each other: the interval of their ratio of long parts holds 1, or the ratio lies within 0.03 of 1.
* **s4 tests** — the test files of both tools pass in the lead's run on the dev box and on the pod before the run of record.
* **s5 the permuted arm** — PM has the k of SP90; the sorted factors of PM and of SP90 on the label rows are equal within 1e-12, and so are those on the other rows with an output.

This card has no map of arms to controls: with one control failed nothing of it is opened, the card is reported as failed with no name, and a changed form is a new card.

## 6. Committed readings

* **PLACED-BETTER-THAN-FLAT and BAR-MET** → the rollout card of section 6 of card ADD132 (SP90 in the lead rule at readout K with no pre-roll on four model seeds, about 0.6 pod-hours).
* **Any other name against the flat margin** → no rollout is spent on the spread; the next card turns to the unsafe rate of the label form itself, as section 6 of card ADD132 states. `SPREAD-MARKS-LONG-READS` beside such a name is stated as it is: the spread carries a mark, and a dose of 10 % placed by it is not better than the flat one.

## 7. Cost and limits

ESTIMATED under 5 minutes of CPU on the fork pod, no GPU time. Limits as card ADD132, section 7; one permutation defines PM.
