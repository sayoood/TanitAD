# PREREG_V8F_SLOWWM, AMENDMENT 2: control 3's clip set and SLOW's cov(u) (Master Mind, 2026-11-11 ~20:47; before any SLOWWM training run or read)

Status: the SLOWWM chain stopped at its basis stage on 2026-11-11 20:44 (`ZZ SLOWWM-STOP basis gate-or-pin-refused rc=4`). The basis was fitted and cached on box 2, but no pilot, training run or read exists. The card, `AMEND_SLOWWM_1.md` and `BESIDE_RULES_SLOWWM.md` are not edited. This file settles two points in card control 3 (card line 73) before the basis stage runs again.

## A. Control 3's identity check runs on all 8 clips (mechanical)
* MEASURED: `theory/ladder/clips8.json` holds 6 eval and 2 train clips. The basis stage ran `slow_basis.py check3 --split eval`, so the identity check saw the 6 eval clips. The box-2 checker requires 8 and refused (`check3 ran on fewer than 8 clips`); the card says "on 8 clips".
* Decision: check 3 runs once per split (eval: 6 clips, train: 2 clips, each against that split's cache), and the stage combines the two records. The identity bound (1e-4) must hold on each of the 8 distinct clips, for both bases, and the wrong-kind basis must fail it on each. All other parts of check 3 are read once, as today.

## B. SLOW's cov(u): the card states two things that cannot both hold
* Card line 26 and `slow_basis.py:152` fix SLOW's normalisation: Vᵀ (C + ridge I) V = I. Then cov(u) = I - ridge Vᵀ V on the fit frames, not I. Card line 73 also asks cov(u) = I within 1e-3 for both bases.
* MEASURED in `CHECK3.json` (box 2, 20:43): PCA-B max deviation 5.6e-8 (meets 1e-3); SLOW max deviation 0.0154, diagonal 0.9846 to 0.9999. `slow_basis.py check3` marked `slow_pass` false, `pass_cov_slow_card_tolerance` false, and its overall `pass` true. That was a silent pass of a failing pre-registered control; it is not accepted as is.
* Decision: **the normalisation stays as the card fixes it.** The fitted basis is the experiment's object, and SLOW and PCA-B differ by at most 1.6 % in per-direction input scale. Rescaling SLOW to unit variance would change `slow_basis.py`, its pinned sha256 and the code of record. For SLOW, control 3's cov check becomes the exact identity the normalisation implies: **cov(u) = I - ridge Vᵀ V within 1e-3** on the fit frames. The deviation from I is reported beside, per direction, in every SLOWWM result. PCA-B keeps cov(u) = I within 1e-3.
* The checker must enforce both forms. A record whose overall `pass` is true while either cov check fails is refused.

## Unchanged
Arms, argvs (`frozen/argvs_slowwm.json`), code of record `f7e790ab`, the deciding read (`AMEND_SLOWWM_1.md`), outcomes, and the beside rules.
