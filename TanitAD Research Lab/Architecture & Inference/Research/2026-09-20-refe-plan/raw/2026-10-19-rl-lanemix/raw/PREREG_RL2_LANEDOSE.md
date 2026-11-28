# PREREG RL-2: dosing ONLY the lane output of the RL scorer (zero training, CPU, RL-val selection tier)

Written 2026-10-18 by the RL-2 Architecture & Inference agent. sha256 of this file in `PREREG_RL2_LANEDOSE.sha256`, written BEFORE `lanedose.py` or `composites_ext.py` scored a single pick against a label. Tier: RL-val SELECTION tier (the true NAVSIM v1.1 sub-scores of the argmax pick on 3,000 held-out RL-val sets / 75 logs), never T1, never an endpoint. Evidence class of every number it will produce: MEASURED (banked live logit dumps + `pdm_val`/`terms_val` labels), one training seed per arm.

## 0. Disclosures (what I had seen before writing this)

* READ from RL-1's `out/RESULT.md` (full-RL-val, R0 u1000, vs the deployed base0, point estimates and CIs as printed there): V1 (= lane dose 0) PDMS +0.17 [-0.12, +0.40], EP +0.64, TTC -0.07 [-0.38, +0.21], NC -0.08 [-0.19, +0.00]; the descriptive D1 row (= lane dose 1.0, same heads) PDMS +0.34 / EP +1.02 / TTC -0.10 (no CI printed); B itself PDMS +0.59 [+0.14, +1.02], TTC -0.17; V1 vs own A: TTC -0.90 [-1.71, -0.14]; the lane Shapley share 0.60. **So the two ENDPOINTS of the family (dose 0, dose 1) are known to me on the FULL set, and the HELD half is part of that set.** The interior doses (0.25, 0.5, 0.75) and the three clips are unseen. The HELD half is therefore "held" with respect to SELECTION, not with respect to my prior knowledge of the endpoints. Nothing about R0b u500 or R0d composites, and nothing about any half-set, has been seen. I did NOT open `mechanism_r0d.json` or `composites_rlval.json` (only the brief's one-line summary of R0d: lane share 0.49-0.62 vs own A, MECH-PROGRESS).
* OUTCOME-FREE facts computed before this file (`stage0_taus.py`, `stage0_taus.json`, sha256 `3b586b83594cda8060aef0be9d1337591d5b56c7f542d364d6dd606ac5ffd7fc`; reads only logit dumps, never a label): the three step-0 base dumps are byte-identical (sha256 `6332fcc6...fef63` for r0, r0b, r0d), so "base0" is ONE array for every arm; B and base0 rows are key-aligned in identical order for all three arms; quantiles of |RL lane logit - base lane logit| pooled over 3,000 x 64 proposals: R0 u1000 q25 0.609, q50 1.359, q75 2.625, q90 3.875, q99 5.047 (R0b u500 and R0d u1000 within +-0.07 of these at q25-q75). Mean |delta| 1.72 logit.
* A dry run of `lanedose.py --dry` (picks only, no scoring) was executed: dose 0 = V1 picks, dose 1 = D1 picks, clip tau 0 = V1, clip tau 1e9 = D1, all bit-exact on all three arms; split FIT 38 logs / 1,561 sets, HELD 37 logs / 1,439 sets. No outcome was computed.
* Known structural fact I commit to using in the forecast (section 7): RL-1 showed B itself PASSES the RL-val bar vs base0 (PDMS lower bound +0.14, no separated adverse) while failing it on navtest. RL-val cannot see the navtest NC/TTC loss. A pass here is therefore NECESSARY, not sufficient, for navtest.

## 1. Code and inputs (hashed)

| item | sha256 |
|---|---|
| `lanedose.py` (step 3; arguments: `--out`, `--prereg-sha`) | `c4220a025be7b2362d897fc528c0fd2a7a5d722d7c3c3d0b3d1340aab2866fc6` |
| `composites_ext.py` (step 2 extension; RL-1's `composites.py` with the diffs of section 6) | `ab5851ce7f50ab3edd508a9a4e4e7d8f6c454ee30d64f4f919a7c12ad0b755ac` |
| `rlh.py` (RL-1's helper, copied unchanged) | `191d97d2f8bd027517cd84c76d8f8cab45d5c398af531b69860e1e434fcffc0b` |
| RL-1 `composites_rlval.json` (read only AFTER this file is hashed, as the reproduction control) | `1ce3b6387f918a65708bedc20224ecef4953610da4665c2952c25121ea8dcdf9` |
| RL-1 PREREG_B / PREREG_A | `0593d7922368c878d7e42c82e1eaf30f2f0be1a399abb4ad7d3471db7eaddf58` / `c7b1c1c59cbd8cf228459bbc3bc61e3b4d8a576fd502f39d6eac06f5f33c7335` |

Dumps (logit space, [3000, 64, 7]; channels 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC, 6 lane; rule `navsim_v1_lane` unchanged), sha256 of the files read: R0 u1000 B `65dcd60e12744717ec535cbbbc07d4aa6ef4f52a8be1438baf98c1ac232aea46`, A `c5986680bd076987e2a9dd39b6368b3df8eaeb32f5afa6d74e243d0e2886ed2f` (`C:/Users/Admin/rls1b_dl/r0/final/scores_update1000_{B,A}.npz`); R0b u500 B `74910863de3a7557afd6daa8c2f655c95a1724c757dfec9346aeec1f02691277`, A `7f5e56e398f1e2034822b20421d99e1fb6e0662ee1665199beddc84a7a559b4c` (`.../r0b/final/scores_update500_{B,A}.npz`); R0d u1000 B `557ed17226e0eb429d4647a09e48c79e13392fdd4c3407e7819cce483c9b459e`, A `e6bc8eb584b5d5d7e3bb78d5f11b3b6e3990484c762990d21ae1b0c5032bb489` (`C:/Users/Admin/par18/rl/r0d_dl/r0d/final/scores_update1000_{B,A}.npz`); base0 `.../r0/final/scores_step0_base.npz` `6332fcc6175805349dd84766d6e38eac79f3ab194588b541755c1cf6281fef63`. Labels: `pdm_val` + `terms_val` through `rlh.load_labels` (unchanged).

## 2. The family (7 selectable configs; at most 9 allowed by the brief)

All configs take EP and C (channels 2, 4) from the RL scorer B, NC, DAC, TTC, DDC (0, 1, 3, 5) from base0, and set the lane logit (channel 6) by:

| config | lane logit | note |
|---|---|---|
| `L0.25`, `L0.5`, `L0.75`, `L1.0` | `(1-a)*base + a*RL`, a = 0.25, 0.5, 0.75, 1.0 | algebraically `base + a*(RL-base)`; this form is exact at both ends. Control: the two forms differ by 0.0 |
| `C0.6`, `C1.4`, `C2.6` | `base + clip(RL - base, -tau, +tau)` | tau FIXED here (not derived at run time) as the R0 u1000 quantiles q25, q50, q75 of |lane delta| (0.609, 1.359, 2.625) rounded to one decimal; the same three numbers are used for every arm |

Not selectable, run as controls/references only: dose 0 = V1 (RL-1's composite), `B_itself` (all channels RL), `own A`. `L1.0` is RL-1's descriptive D1. The three clip grids and the four doses are the complete search space: **7 selectable configs + 1 reference (V1) = 8 <= 9.** No other config will be scored.

## 3. Protocol

1. **Arms:** R0 u1000 (PRIMARY, its row is the formal verdict, as in RL-1), R0b u500, R0d u1000 (consistency arms). own A = the same arm's `scores_update{u}_A`. Comparators: (i) base0 (the navtest bar's comparator), (ii) own A (bar S).
2. **Split:** sorted unique log names (75); `np.random.default_rng(20261019).permutation(75)`; the first 38 permuted positions are FIT, the other 37 HELD. Same split for every arm. Written into the output JSON with the log list.
3. **Estimator:** `rlh.Boot` (paired log-cluster bootstrap, 10,000 resamples, seed 20260927) built on the logs of the half being read (FIT: 38 groups; HELD: 37; the descriptive table: 75). It answers "another draw of LOGS" only; one training seed per arm; H-ESTIM-SEED-1 applies. No inference run (the scorer is deterministic).
4. **Per config and half, all x100, paired:** PDMS, NC, DAC, EP, TTC, C, DDC vs base0; PDMS, TTC, NC, EP vs own A; all seven vs V1 (the lane-dose contrast); PDMS/TTC/NC vs B; retention of B's delta (ratio of means, same resamples; unstable when B's delta is near 0, read with the CI). Families at this tier: LONGITUDINAL = EP, TTC; LATERAL = DAC, DDC; TACTICAL and lateral kinematics NOT COMPUTABLE (n = 0); STRATEGIC PAUSED (PI item 25, 2026-09-19).
5. **Label (RL-1's, vs base0, on the half read):** PASS-VAL = dPDMS lower bound > 0 AND NC, DAC, TTC upper bounds >= 0; GAIN-LOST-VAL = dPDMS lower bound <= 0, no adverse; SAFETY-LOST-VAL = lower bound > 0, some adverse; BOTH-LOST-VAL = neither. **Bar S (vs own A):** TTC and NC point deltas >= -0.5 (RL-1's form).
6. **Selection on FIT, per arm (mechanical, no judgement):** deficit(c) = max(0, -LB[dPDMS]) + sum over {NC, DAC, TTC} of max(0, -UB[dc]) vs base0 on FIT; c* = argmin deficit; ties (within 1e-9) go to the smaller bar-S shortfall (max(0, -0.5 - dTTC_A) + max(0, -0.5 - dNC_A), points), then the higher dPDMS point, then the earlier config in the section 2 order. FIT-feasible set F = configs labelled PASS-VAL on FIT.
7. **HELD is read ONCE, for c* only** (plus three non-candidate references: B itself, V1, own A, vs base0; they exist to give scale and the power flag). The other six configs are never scored on HELD alone. After the verdict rows are fixed in the JSON, a **DESCRIPTIVE table on FIT union HELD** (all 7 configs, V1, B, own A) is written, labelled post-verdict; it never selects, never changes a row. If it disagrees with the formal row, the formal row stands and the disagreement is stated in RESULT.
8. **Power flag.** UNDERPOWERED := B itself has dPDMS lower bound <= 0 vs base0 on the HELD half. (A half-sample PDMS bound is hard to clear: B's full-set bound is only +0.14. A config that retains B's gain would then read GAIN-LOST-VAL on HELD for lack of resolution.) The label is still reported as written and the row still applies; the flag is a statement about the bar, not a goalpost move.
9. **Dose-response diagnostics on FIT (alpha grid only).** G(a) = FIT dPDMS point of L(a) minus V1's; K(a) = min over {TTC, NC} of the FIT point of L(a) minus V1's (negative = a cost). COMOVE_FIT := F empty AND for all four a: G(a) > 0 AND K(a) < 0 ("every dose that buys gain over V1 also buys a TTC or NC cost, and none passes"). FREE_POINT_FIT := some a with G(a) > 0 and K(a) >= 0. NOGAIN_FIT := no a with G(a) > 0.

## 4. Controls (each must hold or the run is INVALID)

1. Pick identity, bit-exact, every arm: dose 0 = V1; clip tau 0 = V1; dose 1.0 = D1 = `compose(B, base0, (2,4,6))`; clip tau 1e9 = D1; algebra difference between the two lane forms = 0.0.
2. Reproduction of RL-1's banked numbers with THIS code and the full-set `Boot`: dose 0 vs base0 must equal `composites_rlval.json` `r0_u1000.V1.vs_base0` and dose 1.0 must equal `descriptive.D1_H1shape_base0.vs_base0` (mean and both CI ends, all seven metrics) to 1e-9.
3. Deliberate-regression arm: `L1.0` with NC, DAC, TTC, DDC taken from the base0 of a fixed permutation of the sets (seed 20261020) on the full set must read dPDMS lower bound < 0 (the bar fails it).
4. Bootstrap identity: `Boot.ci` equals `proxy_eval.boot` (max abs 0.0) on a TTC vector, for the full set and for each half.
5. Split sanity: halves disjoint and complete; log sets disjoint; 38/37 logs.
6. Extension (section 6) controls: all-RL = B and zero-RL = SRC bit-exact for both SRCs; permuted-SRC regression arm; bootstrap identity; and R0 u1000 reproduced against the banked file to 1e-9 with the same script.

## 5. Committed outcomes of step 3 and the next arm of EACH row (primary arm R0 u1000, HELD label of c*)

| row | condition | reading | committed next arm |
|---|---|---|---|
| **R1 L-PASS** | HELD label PASS-VAL and bar S passes | a partial lane dose keeps the RL-val gain inside the bar (necessary, not sufficient, for navtest) | navtest registration of c* (below), then STOP: not built here; a second navtest look needs the caller/PI (RL-1's one look is spent on V1) |
| **R2 L-PASS-BASE** | PASS-VAL, bar S fails | clears the deployed-scorer bar, does not hold own A's TTC/NC (V1 already misses S by -0.90 through the base safety heads, a source effect, not a lane effect) | same navtest registration; the S miss is named, since navtest compares to BASE |
| **R3 L-SAFETY-LOST-HELD** | HELD label SAFETY-LOST-VAL or BOTH-LOST-VAL | the dose that keeps the gain separated carries a separated NC/DAC/TTC cost: lane is one carrier | anchored RL arm (section 8), BLOCKED on A40 compute |
| **R4a L-GAIN-LOST-HELD-COMOVE** | HELD GAIN-LOST-VAL and COMOVE_FIT | gain and TTC/NC cost move together at every dose; lane is ONE carrier, only training can split it | anchored RL arm (RL-1 section 7), BLOCKED on A40 compute |
| **R4b L-GAIN-LOST-HELD-FREEPOINT** | HELD GAIN-LOST-VAL, not COMOVE_FIT, FREE_POINT_FIT | a dose with positive gain over V1 and no point cost exists on FIT but HELD cannot confirm it (read the UNDERPOWERED flag) | navtest registration of the FREE-POINT dose with the largest G on FIT (written, not built); the T1 look is the only discriminating test because RL-val is blind to navtest NC/TTC; needs the caller/PI. If refused: anchored RL arm (BLOCKED) |
| **R4c L-GAIN-LOST-HELD-NOGAIN** | HELD GAIN-LOST-VAL, NOGAIN_FIT | lane dosing buys no gain over V1 on FIT | anchored RL arm, BLOCKED; the gain then lives outside lane+EP/C composites |

The same mapping is applied to R0b u500 and R0d u1000 and reported as consistency. The headline row is R0 u1000's; if a consistency arm lands in a different row the disagreement is stated and the headline stands. A PASS on one arm only is flagged one-arm (no replicate, H-ESTIM-SEED-1).

**What each blocked next arm needs:** the anchored RL arm (3-line design in RL-1 RESULT section 7: lr 0 on the NC, DAC, TTC, DDC, lane slices of `score_head` during EPO, R0's reward and `--epo-lam 1.0`, logit anchor behind a G-DVB-gated flag) is ESTIMATED at 3.7 A40-h per arm (R0d wall clock) plus a control. It is BLOCKED because the A40 runs P6 and the PI declined extra compute; it unblocks when (a) the PI grants about 4-8 A40-h or P6 frees the A40, and (b) a launch-gate profile for the new flag passes (`stack/scripts/launch_gate.py`, G-DVB entry for the new anchor flag, PASS token bound to commit and argv). Cheapest zero-training lever NOT run here because it exceeds the 9-config cap and is a new family: a safety-conditioned lane dose (apply the RL lane delta only on proposals whose base0 NC and TTC pass probabilities exceed a fixed threshold); it needs its own PREREG from the caller.

**Navtest registration (written for R1/R2/R4b; NOT executed):** (i) extend `nt_build.py` `VAR` with the lane composite of c* (lane logit as in section 2, EP/C from R0B, rest from BASE, using the banked chunk logits `C:/Users/Admin/freept/work/out/chunk_00*.npz`; only R0 u1000 has navtest chunk logits, R0b and R0d do not); (ii) gates: BASE and R0B picks reproduced 12,146/12,146, all-RL = R0B, none-RL = BASE, and the dose-0 build equals `points_V1.npz`'s V1 picks; (iii) `nt_run.py` CPU scoring through W3's `score_e.py` with its own `--out` (RL-1 expected 25-60 min; FREEPOINT 1,335.7 s unloaded), marker `DONE_<V>`; (iv) `nt_read.py`: bar dPDMS lower bound > 0 vs BASE and NC, DAC, TTC upper bounds >= 0 (DEPLOYABLE-SAFE), lever view vs R0B and C0A, families6. One navtest look is spent on V1; this second look requires the caller/PI.

## 6. Step 2: RL-1's composites V1/V2 applied to R0d (an EXTENSION, not a new test)

`composites_ext.py` = RL-1's `composites.py` with exactly these diffs (`diff --strip-trailing-cr composites.py composites_ext.py` is stored beside the output): states `R0 u1000` (reproduction control) and `R0d u250/500/750/1000` (u1000 primary, the rest consistency, as RL-1 did for R0/R0b); the R0d dumps are read from `C:/Users/Admin/par18/rl/r0d_dl`; RL-1's prereg hashes read by absolute path; descriptive rows D1-D4 for R0 u1000 and R0d u1000; the original's `git show FETCH_HEAD` machinery check is replaced by a bit-reproduction of RL-1's banked `r0_u1000` block (every number, 1e-9). Everything else, including SRC choices (V1: base0; V2: C0-A at the same update, NOT R0d's own A, the code is identical), labels, deficit and the bar, is RL-1's PREREG_B sections 1-2 unchanged. No navtest, no new selection (RL-1 already selected V1 on R0). own A in the "vs own A" columns is R0d's own A. Next arms per RL-1 PREREG_B section 4: GAIN-LOST-VAL -> anchored RL arm (BLOCKED, section 5); SAFETY-LOST/BOTH-LOST -> the D3 single-head-anchor table (written in the JSON). Forecast: V1 reads GAIN-LOST-VAL at R0d u1000 (as at R0 and R0b).

## 7. Forecast (a prediction, not a bar), committed before any half-set number

* Row family: R4 (HELD GAIN-LOST-VAL) for R0 u1000, most likely R4b, then R4a; R1/R2 improbable (about 10 %). UNDERPOWERED flag true (B's HELD PDMS bound crosses 0). Bar S fails for every config on every arm (the safety source is base0, as in V1). Interior doses move PDMS monotonically between V1's +0.17 and D1's +0.34 on the full set, and TTC changes by less than its CI.
* If that is right, the family answers its own question: lane dosing moves the PDMS point and the TTC point along one line inside noise, and the T1 test (navtest), not RL-val, is what would discriminate.

## 8. Reporting rules

Every table carries tier (RL-val selection), estimator (paired log-cluster bootstrap, "another draw of LOGS" only, one seed) and the artifact path. The formal rows are reported as written; the descriptive table is labelled post-verdict. Proposed GAC/BACKLOG rows go into RESULT.md; no register is edited.
