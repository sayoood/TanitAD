# PREREG: P6-HN step 0 -- expert-proximate hard-negative census on the navtrain held-out table

Written 2026-10-18, BEFORE any census quantity (any distance to an expert, any proximate count, any hard-negative count among proximate
proposals) was computed. Hashed in `PREREG_HN_CENSUS.md.sha256`. Anything not defined here is not a result of this package.
Evidence class of every census number: **MEASURED (descriptive; nothing fitted, nothing tuned)**. Section 7 is **ESTIMATED**.

## 0. Question

LAB-RUN-019 finding 2 / `P6-HN` step 0: per set, how many expert-proximate proposals exist, how many of them are hard negatives
(PDMS zero), and at what distance? If the existing fan already supplies expert-proximate hard negatives per scene, a hard-negative
scorer-loss term can be fed from labelled sets; if not, an offset bank (Lab step 1) is the only supply.

## 1. Data (DEV set; the navtest table is NOT opened)

* Table: navtrain **held-out** sets, 3,137 sets / 24 logs / 64 proposals each; fan = `model_final` ckpt_step 10075 (u400's generator, NOT P2's).
  SEL-RANK measured 0 logs shared with navtest (`raw/2026-10-18-sel-rank/raw/dev_replicate.json` gates). This package never opens the navtest table.
* Labels: `D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_r{0,1}_hw*.jsonl` (nc, dac, ep, ttc, c, ddc, pdms; label_version 5,
  plans "executed (A7 repair)"; file sha256s in SEL-RANK `inputs.json`).
* Proposal poses: `D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets/onpolicy_r*_hw*.jsonl`, field `traj` [64][20][2] (ego frame, x forward, y left, 20 poses,
  stored in float16 resolution). Joined to the labels on the full key (log, token, step, rank).
* Expert, PRIMARY **E_T** = the banked **teacher plan** of the key: `D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/queue/props_r*` field `teacher` [20][3]
  (= `TargetBank.rows[i]['traj']`, `refe/onpolicy_dump.py:141`). It is the labels' REFERENCE candidate on navtrain ("the teacher plan on navtrain; the human future on navtest",
  `eval/PREREG_MEASURE5R.md:28`; EP is normalised to it). The two teacher ranks (rank 0 / rank 1) differ (median 2.4 m, PAPER_CONFORMANCE_REVIEW), so the key INCLUDES rank.
* Expert, SECONDARY **E_H** = the logged **human future** from the nuPlan DB (`D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/**/<log>.db`), ONLY for the logs whose DB is
  present locally (probed 2026-10-18: 13 of 24). Extraction: t0 = `lidar_pc.timestamp` of the key's token; the ego pose at `lidar_pc.ego_pose_token` defines the frame; poses at
  t0 + 0.2*k s, k = 1..20, by linear interpolation of `ego_pose` x, y (read-only sqlite); expressed in that frame. Coverage is reported. **Gate G7:** median ADE(E_T, E_H) over the covered sets
  must be <= 5 m, otherwise the secondary is declared VOID (reported, not used).
* Frame/time: the 20 poses are taken as stored; the 0.2 s spacing is INFERRED (10 Hz sim, stride 2) and only matters to E_H. The 4 s horizon is the stored 20 poses.
* Not available for navtrain (named): the NAVSIM `driving_command` (only the `test` OpenScene metadata is on disk), so the "by driving command" cut is replaced by a pre-registered
  PROXY, the expert-manoeuvre class of section 3.6. It is a label-side derivation (allowed), not the command.

### Gates already run BEFORE this hash (join validity only; `raw/gates_data.json`; no census quantity was computed)

* 3,137 / 3,137 label keys found in the sets and in the teacher queue, 0 duplicates, 0 NaN; rank 0: 1,705 sets, rank 1: 1,432; all step 0, all label_version 5, all ckpt 10075.
* **G3** PDMS == 0 <=> NC*DAC == 0 on all 200,768 proposals (0 disagreements). NC takes {0, 0.5, 1}; DAC {0, 1}.
* **G2 order alignment:** the sets' own `navsim_dac.violation > 0` equals the labels' `dac == 0` on **200,768 / 200,768** proposals (100 %), against a within-set permutation control of 90.3 %
  (max 90.4 % over 20 permutations): the label arrays are in the same proposal order as the stored poses.
* Disclosure: while reading the schema I saw ONE set's first proposal pose and first teacher pose (both ~ (0.59, 0.00)); no distribution or count was seen.

## 2. Estimator

Log-cluster bootstrap over the 24 logs, **B = 10,000 resamples, seed 20260927**, percentile 2.5 / 97.5, ratio estimators sum(num)/sum(den) with the SAME draws everywhere
(`zero_tail.Boot` semantics, re-implemented in `code/hn_census.py`, tested against a brute-force resample). It answers **"another draw of LOGS"** only. 24 clusters is small:
the intervals are wide and a log-level driver (a few logs with many hard negatives) is visible in the per-log table. It does not answer another training run or another fan.
One fan (ckpt 10075), one teacher rollout per key, no seeds: **nothing here is a lever effect**.

## 3. Definitions (fixed)

3.1 **ADE(p, E)** = mean over the 20 poses of the Euclidean xy distance between proposal p and the expert E. **FDE(p, E)** = distance at pose 20 (4.0 s). ADE is the PRIMARY proximity;
FDE is a sensitivity reading with the same grid.

3.2 **Expert-proximate at d**: ADE <= d, d in the committed grid **{0.5, 1, 2, 4} m** (inclusive). The grid is reported whole; no single d is picked after seeing data.
The decision anchor of section 6 is d* = 0.5 m (the Lab's step-1 offsets go up to 0.6 m).

3.3 **Hard negative**: PDMS == 0, i.e. NC == 0 or DAC == 0 (G3). Sub-types: DAC-only (DAC 0, NC > 0), NC-only (NC 0, DAC 1), both. NC == 0.5 (static object) is NOT a hard negative (PDMS > 0).

3.4 Per set at each d: n_prox(d) = #proposals expert-proximate, n_hn(d) = #of those that are hard negatives.

3.5 **Twin pair (Lab step 0)**: two proposals i, j of one set with DAC_i = 0, DAC_j = 1 and mean-over-20-poses pairwise L2 distance <= t, t in **{0.3, 0.5} m**. Mean L2 >= mean lateral gap, so
this is conservative ("same manoeuvre" is implied by staying within t over 4 s). Generalised pair: PDMS_i = 0, PDMS_j > 0 (same t).

3.6 **Expert-manoeuvre class** (from E_T only): path length L = sum of consecutive xy steps starting from the origin; STATIONARY if L < 2.0 m; else LEFT if final yaw >= +15 deg, RIGHT if <= -15 deg,
else STRAIGHT (15 deg = ZT3's `TURN_DEG`). Sanity gate G8 (run after this hash, reported): median final y of LEFT > 0 and of RIGHT < 0; if the sign is inverted the class labels swap and this is reported.

## 4. What is reported (all with the section-2 CI unless marked "point")

**A1 -- supply, per d in {0.5, 1, 2, 4}, ADE and FDE:** share of sets with n_prox >= 1; **share of sets with n_hn >= 1 (the headline)**; that share among sets with n_prox >= 1; mean n_prox and mean n_hn per set;
pooled hard-negative rate among proximate proposals (sum n_hn / sum n_prox); the same headline split by sub-type (DAC-only, NC-only, both); the histogram of n_hn per set (0, 1, 2, 3-4, 5+; point).
**A2 -- distances** (exact 0.01 m histograms, quantile by in-bin interpolation, CI by resampling the per-log histograms): quantiles 5/25/50/75/95 of (a) ADE of all 64x3,137 proposals, (b) the nearest-proposal ADE per set,
(c) the nearest-hard-negative ADE per set (sets with >= 1 hard negative anywhere in the 64; their share is reported), (d) ADE of hard negatives vs non-hard-negatives pooled; and **(e) the hard-negative rate by ADE band**
[0,0.5], (0.5,1], (1,2], (2,4], (4,inf) and the same by FDE band. Also the share of ALL hard negatives that are expert-proximate at each d.
**A3 -- strata:** the A1 headline (s_hn) at each d by expert-manoeuvre class and by rank (0 / 1); sets and logs per stratum; a stratum with < 100 sets or < 5 logs is point-only.
**A4 -- per-log table** (point): sets, hard-negative rate overall, s_hn(d) per log, so a log-driven result is visible.
**A5 -- secondary expert E_H** on the covered logs (G7 permitting): A1 and A2(e) with E_H, and E_T recomputed on the SAME covered sets for a like-for-like comparison.
**B -- Lab step 0 twin pairs:** share of sets with >= 1 DAC0/DAC1 twin pair at t = 0.3 and 0.5; mean pairs per set; **the twinned fraction** = share of DAC-0 proposals that have >= 1 DAC-1 twin within t; the generalised (PDMS) version.
**D -- distributions of the sets themselves:** hard-negative rate overall and per set (quantiles), so the census has its base rate.

## 5. Controls and tests (run before the result is read; RED mutants must fail)

* **K1 identity:** with E := proposal 0 of each set, n_prox(d) >= 1 and nearest ADE = 0 on all sets, all d.
* **K2 wrong expert:** with E := the teacher of a DIFFERENT set of the same log (cyclic shift within log, E-class unchanged in distribution), the nearest-proposal ADE median must be larger than with the true teacher; the
  true-expert A1 headline must exceed the wrong-expert headline (the census is reading the expert, not the fan geometry). Both are reported next to A1.
* **K3 brute force:** the bootstrap on a 3-log toy equals an explicit resample loop to 1e-12; the histogram-quantile equals `np.quantile` to the bin width on a toy.
* **K4 mutants:** (i) `<=` -> `<`; (ii) ADE computed over only pose 0; (iii) hard negative defined as DAC==0 only; (iv) rank dropped from the join key. Each must turn a named unit test RED.
* Tests: `code/test_hn_census.py`, run with the driverl-eval interpreter, output in `raw/test_hn_census.log`.

## 6. Readings committed in advance (descriptive aids; they never change an evidence class)

Anchor d* = 0.5 m, ADE, E_T; classified on the POINT estimate of s_hn(0.5) = share of sets with >= 1 expert-proximate hard negative. If the 95 % CI straddles a boundary the reading is "straddles" and both sides are reported.

| s_hn(0.5) | reading | consequence for P6-HN |
|---|---|---|
| >= 0.50 | **SUPPLY EXISTS** | step 1 can mine the existing labelled on-policy sets; the offset bank is an add-on |
| 0.10 .. 0.50 | **PARTIAL** | mine what exists AND build the offset bank for the scenes with none |
| < 0.10 | **SPARSE** | the existing fan does not carry the term; the offset bank is the only supply and step 1 is the whole lever |

Lab's own rule for part B: if the share of sets with >= 1 DAC0/DAC1 twin pair at t = 0.5 is < 0.10, "the bank is where the signal is missing"; otherwise pairs are present per set.
A1/A2 are read at ALL d regardless; if the headline is non-monotone in d that is reported as such.

## 7. Reach estimate (ESTIMATED -- arithmetic on the census, not a measurement)

Inputs, each with its class: **kappa = 0.953 PDMS per pp of zero rate** (= 4.15 PDMS / 4.36 pp; MEASURED inputs, `D-REFE-GAP-CI`, 529 / 12,146 navtest u400; carried in LAB-RUN-019 section 4.3);
**Z = 4.36 pp** (MEASURED, navtest u400; INHERITED here, not re-measured); published relative cuts **r = 0.208** (DA-WAM `2608.19085` Tab. 3) and **r = 0.227** (BeyondDrive `2605.19771` Tab. 4, on LTF*) (PUBLISHED, single runs).
* **C1 supply-limited reach(d) = s_hn(d) * r * Z * kappa** [PDMS], at the census' s_hn(d) with its CI endpoints. ASSUMPTION (HYPOTHESIS): the published methods supply a hard negative to every scene and the effect scales linearly with the
  share of scenes served. The Lab's unconstrained standalone value is r*Z*kappa = 0.86 to 0.94.
* **C2 zero-pick coverage:** for the three SFT-4 sibling scorers N-s0/1/2 (`s1_read_cache/arms/N-s*/dump/scores_update400_B.npz`, rule navsim_v1, NOT P2's scorer; each READ SEPARATELY, no pooling): the share of
  zero-PDMS PICKS that are themselves expert-proximate at d, and the share of zero-pick sets that contain >= 1 expert-proximate hard negative. **Ceiling(d) = coverage * Z * kappa.** A ceiling, not an expectation.
* Everything in section 7 is on the dev table's fan and sibling scorers and is transplanted to navtest units through kappa and Z: that is the ESTIMATED step. Gross, not net (`D-REFE-ZEROTAIL-4` warns a gross reach can net ~0).

## 8. Out of scope (named)

No scorer is trained, no label is created, no P6 arm is touched. The navtest table, the pod and the GPU are not used. The unlabelled training scenes' fan and the offset bank do not exist yet; this package says
what the existing held-out table shows, not what the full training set would.
