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


