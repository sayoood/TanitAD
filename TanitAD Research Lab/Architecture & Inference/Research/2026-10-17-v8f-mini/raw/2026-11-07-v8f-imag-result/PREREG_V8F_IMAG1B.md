# PREREG_V8F_IMAG1B: the Imagination Audit with a refit permutation reference and a working real-vs-real control (Master Mind, 2026-11-07, written after IMAG1's INVALID control read and before any IMAG1B code or statistic; no IMAG1 label or result row has been read by anyone)

## Why
IMAG1 (`PREREG_V8F_IMAG1.md` `14846eef`, IMPL_CHOICES `2ecd212f`) is **INVALID** on eval-141 by its controls 2 and 3, as committed. The measured causes come from the control rows of `theory/imag1/RESULT_IMAG1.md` only:
* **Control 3: the paired sign-flip reference is not a null for these rows.** On 20 draws with permuted labels, the two-part rule separates 18.42 % of LON rows, 24.12 % of LAT rows and 23.71 % of SCENE rows (at most 5 % allowed). Almost all of them are "predicted vs Copy-Last" rows (LON 2,058 of 3,600; LAT 1,046 of 1,440; SCENE 569 of 720) and gain rows (LON 219 + 186 of 720 per side; LAT 237 + 222). A probe fitted on permuted labels still reads a fixed latent direction. Predicted and copied latents differ along that direction, so their difference is consistently signed even with no label signal, and a sign flip centred on 0 cannot see this. The IA-1 and Oracle rows calibrate (LON 1 of 1,200 and 3 of 3,600). It is the LATSEM2 failure in a new place, and the fix is the same: **refit the probe under permuted labels and use the statistic's own permutation distribution as the reference.**
* **Control 2: a window 0.2 s off the future is not detectably "not the future".** In 75 of 75 pooled family cells, the encoded latent 0.2 s away read the target as well as the true future. Most targets change little in 0.2 s, so the control asked a question the instrument cannot answer. It must use a not-future that differs in content.

⛔ The label tables and per-row values in `theory/imag1/RESULT_IMAG1.md` and `RESULT_IMAG1.json` stay unread. IMAG1B recomputes its deciding statistics; it may reuse IMAG1's cached contexts, labels, latents and permuted-label draws.

## What is unchanged
Everything in IMAG1's card and IMPL_CHOICES `2ecd212f` that this card does not change stays as written: data, models, targets, probes, folds, horizons, the modules IA-0 / IA-1 / IA-2 / IA-3 / IA-6, the labels, the bars and the triggers. That includes the added labels PROBE-DEPENDENT, SEEDS-DISAGREE and UNRESOLVED, the pooled-bank reading, the TTC cap at 10 s and curvature = kappa0.

## The changes (the only ones)
1. **One reference for every row: refit under permutation.** The permutation reference for every row (IA-1, IA-2 Oracle, predicted vs Copy-Last, IA-3 gain and cosine, IA-6) is the distribution of that same statistic over R = 199 draws. In each draw the labels are deranged across episodes inside the fit fold (LATSEM2B's procedure), the probe is refitted, and the whole statistic is recomputed: for a paired difference, both of its terms with the refitted probe. A row is separated on the claimed side when BOTH hold: its bootstrap 95 % interval excludes the null on that side, AND its point lies beyond the 97.5th percentile (or below the 2.5th) of this reference. The paired sign-flip reference is not used.
2. **Control 2 (real-vs-real), replaced.** The not-future is the encoded latent at t + k of a window from a DIFFERENT episode (uniform draw, seed 0). Requirement: in every pooled family cell where the Oracle is readable (not UNREADABLE), the Oracle's NMSE is two-part separated below this not-future's, at both seeds. Beside it, with no bar: same-clip offsets of +0.2, +0.8 and +1.6 s, reported as a decorrelation curve.
3. **Control 3 (calibration)** as IMAG1: 20 extra draws (seeds 1001 to 1020), each scored against the change-1 reference as if real. At most 5 % of rows may be separated per family, AND per row kind (IA-1, Oracle, predicted vs Copy-Last, gain).
4. **Memory:** at most 4 worker processes (IMAG1's TRAIN beside pass ran out of memory at 7 workers on the shared box). TRAIN stays beside and decides nothing. If it cannot run, it is reported as NOT RUN with the reason.

## Controls 1 and 4
As IMAG1. Control 4 (the synthetic planted effects) is extended: a latent that differs from the present along a fixed direction unrelated to the label must NOT come out IMAGINES under the change-1 reference. This is the failure that invalidated IMAG1, planted.

## Implementation choices
The implementer writes `theory/imag1b/IMPL_CHOICES_IMAG1B.md` (the deltas only: how each row's statistic is recomputed per draw, which IMAG1 caches are reused and their sha256, the not-future draw, the worker count and the projected wall time) and records its sha256 **before any statistic is computed**.

## Log
* 2026-11-07 15:55:43 frozen: prefix 4634 bytes, sha256 40500d68d344e56d5321d4099ea67db8da92a722a62417e1ac179fb2bc68d593 (everything above this line). No IMAG1B code or statistic existed at this time.
