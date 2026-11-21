# AMEND_OBD_SEED3: how the third-seed BAND read combines labels (Master Mind, 2026-11-11, written after the first OBD read and before any seed-10 / seed-11 run exists)

**Trigger.** `VERDICT_OBD.md`: BAND reads SEEDS-DISAGREE (s0 BAND-ADDS, s1 BAND-COSTS). Card line 48 commits *"the read is repeated with a third training seed per init (`--seed 10` / `11`) for the arms concerned before any outcome is quoted"*. The card does not say how four per-seed labels combine. This file fixes that rule before the data exists. The card itself is unchanged.

**DETACH is settled:** DETACH-KEEPS at both seeds (card line 40: both seeds agree). It is not re-read.

## Arms (the BAND question's arms; argv = the banked arm's argv with ONLY `--seed` and `--out` changed)
| tag | from | `--seed` | init |
|---|---|---|---|
| OBD-CTL_s10 | OBD-CTL_s0 | 10 | M2b s0 |
| OBD-BAND_s10 | OBD-BAND_s0 | 10 | M2b s0 |
| OBD-CTL_s11 | OBD-CTL_s1 | 11 | M2b s1 |
| OBD-BAND_s11 | OBD-BAND_s1 | 11 | M2b s1 |

ESTIMATED about 40 min each on box 2, about 2.7 h in total (MEASURED per-run time from the OBD chain log).

## Read
* The same reader (`read/obd_read.py`), the same 24 cells, the same floor G (OBD-BAND_rep vs OBD-BAND_s0), the same estimator.
* Each new pair (BAND_s10 vs CTL_s10; BAND_s11 vs CTL_s11) gets a per-seed label: BAND-ADDS, BAND-NEUTRAL or BAND-COSTS, by the card's rules (lines 36-47) verbatim.

## Combination rule (fixed now)
* There are four per-seed labels: s0, s1 (already read), s10, s11.
* **The band outcome is the label held by at least 3 of the 4, with at least one of them at each init** (M2b s0 and M2b s1).
* Otherwise the outcome is **BAND-UNSETTLED**.

## What each outcome triggers (the card's lines 51-52, applied)
| band outcome | trigger |
|---|---|
| **BAND-ADDS** | OBD-BAND's argv becomes the O/T recipe of record. |
| **BAND-NEUTRAL** or **BAND-COSTS** | OBD-CTL's argv becomes the recipe of record. The band loss leaves the recipe, and its operative gain is recorded as not carried. |
| **BAND-UNSETTLED** | OBD-CTL's argv becomes the recipe of record, the conservative branch: four training runs could not show the band loss consistently. The band's operative-only gain stays recorded beside, with the code-drift caveat (`VERDICT_OBD.md` §3). |

## Beside (no bar)
* The four labels.
* The per-cell spread across the four seeds.
* The code-drift row's sizes, for scale.
