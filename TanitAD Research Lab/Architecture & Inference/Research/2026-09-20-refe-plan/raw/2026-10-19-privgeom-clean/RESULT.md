# PRIVGEOM-CLEAN: RESULT (Master Mind, 2026-10-20 ~01:50)

**Final call: INCONCLUSIVE** (sealed `PREREG_PRIVGEOM_CLEAN.md` sha256 `148ebf0f…`, plus the operational Amendment OPS1 sha256 `3d84a135…`). On the 76 held-out navtrain logs (10,867 sets, 6,140 tokens), adding the centred within-set privileged-proximity block to DAC's learned row raises fast-band within-set AUC by about +0.010 on both seeds. The lower CI ends sit just above 0, but the point is below the +0.02 materiality bar, so the call is neither CONFIRM nor NULL.

Stamps: MEASURED, held-out logs (navtrain, never in SAFE's 41), SELECTION tier (a label lookup, not driving), paired log-cluster bootstrap B = 10,000, seed 20260927 ("another draw of LOGS" only), two fit seeds.

## Controls first (all from `raw/read/results/results.json`)
- **Gate:** PASS.
  - C-JOIN: 10,867 / 10,867 sets and 76 / 76 logs retained, lost 0.0.
  - C-ID-EXTRACT, C-GRAD-REAL, C-ADMIT (the 9-column sign-clean block is admissible) and C-REACH (min 0.99) all pass.
  - The label identity holds at 1.0 on all three clauses (63,915 DAC-0 proposals).
- **C-NULL** (within-set permutation, 5 replicates): 0 / 5 fire. The replicates read −0.0009 to −0.0003.
- **C-POWER** (plant sigma 1.25, calibrated to +0.0276 on the SEEN arrays): reads **+0.0159 [+0.0071, +0.0257] / +0.0157 [+0.0070, +0.0256]**, so it does NOT fire CONFIRM on either seed.
  - By the sealed precedence, a power failure converts only a NULL call (to UNDERPOWERED), so the INCONCLUSIVE stands.
  - It does say the instrument on this population attenuates a +0.028-sized plant to about +0.016. The read was powered for the SAFE reading's size, not for half of it, as the prereg's §8 stated before the read.

## The read
| seed | F-AUC L | F-AUC L+G (primary LGc) | ΔF primary | call | companion LGu ΔF | companion call |
|---|---|---|---|---|---|---|
| 0 | 0.8985 [0.8672, 0.9285] | 0.9092 [0.8793, 0.9377] | **+0.0107 [+0.0004, +0.0252]** | INCONCLUSIVE | +0.0078 [−0.0043, +0.0197] | NULL |
| 1 | 0.8987 [0.8673, 0.9285] | 0.9088 [0.8792, 0.9375] | **+0.0101 [+0.0001, +0.0244]** | INCONCLUSIVE | +0.0075 [−0.0044, +0.0193] | NULL |

- ΔQ1-3 (the slower bands): +0.0056 [+0.0024, +0.0097] / +0.0057 [+0.0022, +0.0099].
- Geometry alone (centred G-only) minus the learned row on F: −0.0729 [−0.0990, −0.0452]. Proximity is a small COMPLEMENT to the learned row, never a substitute.
- The companion (uncentred) is NULL on both seeds. It has no authority over the call (MM ruling DQ1).

## Descriptive only (the committed INCONCLUSIVE report; NO authority)
An inverse-variance combination with SAFE's seen 41 logs (centred +0.0352 [+0.0194, +0.0495], post-hoc-selected) reads +0.0206 [+0.0110, +0.0302] (seed 0) and +0.0200 [+0.0106, +0.0295] (seed 1). It uses normal-approximation SEs from the reported intervals. The 41 carry the selection that motivated this arm, which is why this number decides nothing.

## Four families
- LONGITUDINAL: not computed at this tier (a DAC row read), n = 0.
- LATERAL: DAC is the lateral-bearing row, so this whole read is lateral (drivable-area compliance); heading / curvature / yaw-rate are not computed, n = 0.
- TACTICAL: not applicable, n = 0.
- STRATEGIC: PAUSED (PI ruling 2026-09-19, ITEM 25).

## Committed next arm (prereg §6, INCONCLUSIVE row; RULE ZERO)
- **SH5 stays registered but UNSUPPORTED by this arm.** The held-out proximity signal beyond the learned DAC row is about +0.01 fast-band AUC: real on both seeds (lower CI > 0) but below the +0.02 materiality bar. SH5's sizing (+0.03) is not supported, and its expected value is small next to the search lever (`D-REFE-L2-STEPB-1`, +0.327 PDMS on carve B).
- **First lever: PRECISION, i.e. more LOGS.** More fit seeds cannot buy it, because the interval answers "another draw of logs". The route is an S0 build over more nuPlan logs (compute + corpus): a named blocker needing a PI decision. It shares the S0 frame blocker of `EPADM-SEARCHBANK-S0` (76 logs' frames, 4.9 GB on the pod).
- **MM ruling on priority:** the SH5 C-REACH diagnosis (grid 5, running) is allowed to finish, because it answers an INSTRUMENT question that is already paid for. SH5's LOCAL / FULL runs are DEPRIORITISED behind the search line's GPU needs (C1, the S0 bank, navtest) until a larger effect is measured.

## The operational stop and its fix
The first attempt stopped at 01:09 with `extraction worker crashed 4 times` after the extraction had COMPLETED (76 / 76): the controller compared the venv launcher's pid with the interpreter's pid.
- Fixed by the sealed OPS1 amendment (only worker bookkeeping changed; the extractor and its 76 outputs are unchanged).
- The read resumed at 01:26 and finished at 01:37.
