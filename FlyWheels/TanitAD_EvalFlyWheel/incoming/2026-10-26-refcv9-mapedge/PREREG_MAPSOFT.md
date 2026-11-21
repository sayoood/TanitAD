# PREREG_MAPSOFT -- a SOFT, cross-fitted map cost on refcv8's fan instead of a hard filter (Master Mind, 2026-10-26 ~01:45 CET)

Sealed BEFORE any MAPSOFT pick is computed. 0 training of the network, 0 GPU. The rest of MAPSEL's machinery is reused unchanged
(`refcv9_mapsel/code/mapsel_score.py` as amended by AMENDMENT 2, imported via a wrapper exactly like MAPEDGE Q2).

## 0. Why (MEASURED, already landed / quoted from the two FAIL verdicts that motivate it)
* MAPSEL s0 (FAIL): BASE's emitted plan touches a GT SIDEWALK cell on **17.9 %** of S6c windows (HUMAN 1.25 %). With the TRUE map a
  hard filter (GT-T1) cuts that to 1.7 % and its ADE/FDE are separated BETTER, but it fails C2 (false-decel) and C4 (path 0.67 m
  shorter than BASE, whose paths are themselves longer than the human's). With the OWN map the hard rule (OWN-T1) moves sidewalk only
  -1.3 pp and costs ADE: its plan-level false-alarm rate on HUMAN paths is **18 %** (82 % of human paths pass), so it discards good
  candidates. Template next lever: a learned selector with map-derived labels, not a rule.
* MAPEDGE Q2 (FAIL E2, E4; E1 PASS): the own drivable boundary as a hard tier-2 filter cuts GT edge contact -0.0331 [-0.0483,
  -0.0187] (deranged control +0.0650), but inherits OWN-T1's driving cost. Its committed next lever: a soft cost, not a filter.

## 1. The arm (deployable at inference: uses ONLY the model's own scores and own map)
Per window, over MAPSEL's E9-eligible candidates `c` (same fan, same eligibility, same shipped stage):
`u(c) = z(c) - lam_s * pswmax(c) - lam_e * bnd(c)`, pick = argmax `u` (ties -> lower index), where
* `z(c) = (sel_score_v3(c) - max_elig sel_score_v3) / sd_elig(sel_score_v3)` (sd floored at 1e-6);
* `pswmax(c)` = MAPSEL's banked max own P^(sidewalk) over the candidate's in-range FP5 points (AMENDMENT1 A1.1 plane);
* `bnd(c)` = MAPEDGE's OWN-BND contact flag (any FP5 point within 0.3 m of the own drivable boundary), as banked by the Q2 run.
* `(lam_s, lam_e)` on the grid `{0, 0.25, 0.5, 1, 2, 4, 8}^2` (49 points). `(0, 0)` reproduces BASE exactly (control).

## 2. Fitting (privileged LABELS, cross-fitted; the scored episodes are never tuned on)
* Folds: the 139 eval episodes, permuted with `numpy.random.default_rng(0)`, assigned to 5 folds by rank mod 5.
* For held-out fold k, `(lam_s, lam_e)` is chosen on the OTHER four folds' windows that are S6-compliant, have a GT map and a valid
  human path: minimise `V = mean(gt_off[pick] OR gt_edge[pick])` (MAPSEL's banked per-candidate GT flags of the RAW candidate)
  subject to `H(lam) <= H(0, 0)`, `H` = mean over the 8 slots of the L2 distance of the raw picked candidate to the HUMAN path;
  ties -> smaller `lam_s + lam_e`, then smaller `lam_s`. If no grid point satisfies the constraint, `(0, 0)`.
* The fitted `lam` is applied to fold k's windows only. Labels (GT map, human path) are privileged and used ONLY to fit two scalars on
  other episodes; inference uses own scores + own map.
* **DER-MAPSOFT-XF** (deranged control): identical procedure with the DONOR's planes (`dpswmax`, `derbnd`), fitted the same way.

## 3. Evaluation (MAPSEL's pipeline, unchanged): the EMITTED staged plan vs the GT raster, S6c (+ S2c for the four families); paired
episode-cluster bootstrap B 2000 seed 0 -- "another draw of EPISODES" only; one training seed; one inference seed (0). Tier: T1
self-action OPEN LOOP on logged eval139 frames (not driving performance).

## 4. Controls (any failure VOIDS)
* **S-K0** every MAPSEL hard control passes in this run; MAPSEL's own arms reproduce `refcv9_mapsel/result/s0/MAPSEL_RESULT.json`
  (S6c rates and `vs_BASE` cells) exactly.
* **S-K1** the `(0, 0)` arm's picks equal BASE's picks on every window.
* **S-K2** every episode is in exactly one fold, and no held-out window is in its own fold's fitting set.

## 5. Bars (MAPSOFT-XF vs BASE)
* **M1** GT sidewalk (off-drivable) rate separated BELOW BASE.  **M2** GT edge-contact rate separated BELOW BASE.
* **M3** = MAPSEL C2 (no four-family row, S6c and S2c, separated worse).  **M4** DER-MAPSOFT-XF NOT separated below BASE on M1's or
  M2's metric.  **M5** = MAPSEL C4 (STOP guard, as sealed).  **PASS = M1-M5.**
* Reported beside (no bar): the per-fold `lam`; lane-crossing; the share of GT-T1's sidewalk gain captured; a HUMAN-RELATIVE path
  length line `|len - len_human|` vs BASE (paired) -- descriptive only, it does not replace M5.

## 6. Committed next step
* **PASS** -> register MAPSOFT-XF as an opt-in inference row (T1 stamp); refcv9 trains a map-feature selector/critic with these
  map-derived labels (the soft cost is its initialisation); NavSim only with a PI go.
* **M1 or M2 fails because the fitted `lam` is (0, 0) in >= 3 folds** -> own-map precision binds (the constraint cannot be met):
  the lever is perception training (sidewalk / drivable-boundary precision loss on the map head) -- compute, so a PI decision.
* **M1 or M2 fails otherwise** -> the 2-parameter cost is too weak: the learned listwise selector over richer own-map features.
* **M3 or M5 fails** -> the trade-off costs driving at the fitted point: constrain ALL four families in the fit (multi-objective)
  before anything else.
