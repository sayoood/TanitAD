# ERRATUM-1 to `PREREG_REFCV6_V2.md` — two numbers I quoted without checking them

**Date:** 2026-09-17, hours after the pre-registration landed in `cc24073` ·
**Author:** Master Mind · **Status: BINDING. It corrects the parent file; where they differ, this wins.**

⛔ **Both errors are mine and both are the same failure:** I carried numbers from a session summary
into a pre-registration instead of re-reading them from the record. A pre-registration is exactly
the document where that is least acceptable, because its numbers become the reference the arms are
judged against. Neither error changes a single criterion — the bars, the arms and the refusals all
stand — but the stated evidence behind two of them was wrong.

⭐ **What was CHECKED and is CORRECT, so the reader knows the audit was not selective:** the T-FLIP
figure. `refcv5-v2` **follows a flipped junction command on 20.5 % of windows**, and the acceptance
bar of ≥ 0.50 with true-minus-shuffled ≥ 0.38 (half the measured 0.76 ceiling) is quoted correctly.

---

## E1. §6 — "42.7 % of the TURN label's entropy is already in the nav token" is **UNSUPPORTED. WITHDRAWN.**

**There is no such measurement.** The only `42.7 %` in the steering record is an unrelated statistic:
*"256 of v1's 600 parity-val clips (42.7 %) are inside v2corpus's 9,000-clip training selection"* —
a corpus-overlap figure, not an entropy, not about nav, and not about turns. I attached a real number
to a different claim, which is worse than having no number at all: it reads as measured.

⇒ **Replace it with what WAS measured**, which is stronger than the figure it replaces:

| MEASURED | source |
|---|---|
| `refcv5-v2`'s tactical **turn recall 0.475 → 0.000 with nav removed** — the head is a **pure nav echo** | `REFCV6_CLARIFICATION.md` §3 |
| on the refav1 tactical decoder, under `nav_zero` the ranking **collapses to 0.520**, and a **NAV-ONLY predictor BEATS the model (0.684 > 0.650)** | `D-REFAV1-TAC-DECODER-PANEL`, 2026-09-03 |

⚠️ The second row is **refav1's** decoder, not refcv5's — a different model, quoted as corroboration
of the mechanism and never as a refcv5 measurement.

⭐ **The design consequence is UNCHANGED and now rests on evidence**: turn classes are excluded from
T-ZERO by design, and T-ZERO is scored on the non-turn behaviours. A head whose turn recall goes to
**exactly 0.000** without nav is not partially nav-driven — it is entirely nav-driven, which is a
better reason for the carve-out than the number I invented.

## E2. §5 — the oracle ceiling `0.4713` at 16×40 is **SUPERSEDED**. The value is **0.4762**.

`E-READOUT-CEILING-1` was **partially retracted** on 2026-09-08 (`R-2026-09-08-wpa-mirror`: the
azimuth address in `s5_indexed.py` is mirrored) and the oracle ladder was **re-read under the
corrected address**. Its absolute values are quotable again, and they are not the ones I used.

| rung | **corrected** (tie-group AP, seed 0) | what I wrote |
|---|---|---|
| 16 × 40 | **0.4762** | 0.4713 ⛔ |
| 8 × 20 | **0.3341** | 0.3341 ✅ (correct) |
| `pos_only` control | 0.0316 | — |

⚠️ **And a caveat that matters more than the third decimal:** the **replicate floor is 0.0122 AP**,
and a pure seed replicate — same flags, same address, seed only — reads **+0.0115 "separated"** at
16×40. ⇒ **differences below ~0.012 AP on this ladder are noise**, so the gap my §5 actually depends
on (0.4762 − 0.3341 = **0.1421**) survives with room to spare, and the 0.0049 I had it wrong by is
itself inside the noise floor. The conclusion — *a head on the 160 stride-32 tokens is capped below
the 0.60 bar before training starts* — is unaffected.

⛔ **`SPEC_REFCV6_V2.md` §2 and §6 carry the same stale 0.4713** and are corrected by this erratum
too; they are not separately edited, so that a reader who follows either file's citation arrives
here.

## Why this was caught

By re-reading the record to verify my own landed text, not by anything failing. ⚠️ **Three other
landings tonight quoted numbers I had re-derived from source; these two I did not.** The rule that
would have caught it is one this programme already has — *every number carries its evidence class
and its path* — and a summary is not a path.



---

## E3. §0 — "a measured **14.3 %** one-seed floor" is RIG-SPECIFIC and quotes the MILDEST number we have. **It should be 14.3 %–55.6 %, and the worst case is not a seed effect at all.**

*(Added in the same audit pass, after E1 and E2, by continuing to check rather than stopping at the
first two.)*

**What the parent file says** (§0, standing caveats): *"one-seed separation carries a measured
**14.3 %** floor"*.

**14.3 % is real** — `CLAUDE.md` records it for the **v7-tiny rig**: `separated` differences on
**6 of 42 family cells**, re-derived 2026-09-06. ⛔ **But it is the mildest figure in the record, and
quoting it alone makes the caveat about four times weaker than the evidence supports.**

⭐ **MEASURED on the WP-D rig, and it is worse in kind, not just in degree:** `D0b` — **D0's flags,
D0's SEED, ZERO levers moved** — was itself *separably worse* than `D0` on **5 of the same 9 metrics
(55.6 %)**, reproducing the headline ADE **+0.02610 at +0.02460** and **exceeding the lever on all
three longitudinal metrics**. The two checkpoints went through the identical T1 arm, 3,422 windows /
40 episodes, with `eid` / `window_start` / `gt` / `v0` element-wise identical; mean
`|pred(D0b) − pred(D0)|` = **0.1059 m**.

⇒ **The correction is not a bigger number, it is a different claim:**

| as written | as it should read |
|---|---|
| a **14.3 %** one-seed floor | a **14.3 %–55.6 %** replicate false-positive rate, **rig-dependent**, ⛔ **and refcv6's own rig is UNMEASURED** |
| attributed to **seed** variation | ⛔ the 55.6 % case moved **no seed and no lever** — it is **same-seed replicate noise**, which no seed-control arm would catch |

⭐ **This makes §0's rule STRONGER, not weaker.** "A separated interval is necessary, not sufficient"
was already the rule; what changes is that the honest way to establish a refcv6 result is a
**replicate at the same seed**, not merely a second seed — and that **the floor for these arms has
not been measured and must be**, rather than inherited from whichever rig gives the friendliest
number.

⛔ **And this is the same error as E1 and E2 in a third costume:** I quoted a figure that exists,
from a rig that is not the one under test, without the range or the provenance beside it. It is the
`true-but-wrong-for-the-reader` class — every word defensible, the conclusion the reader draws
wrong.
