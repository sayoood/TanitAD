<title>TF-2 pre-registration</title>

# PREREG TF-2: policy boundary margin (L2b) and DAC-certified labels (L3) on held-out navtrain

Written 2026-11-11 by the TF design agent, **before any TF-2 number exists**. The sha256 of this file is recorded in
`RESULT_TF2.md`. Nothing here may change after the first TF-2 row is written; any later change is an amendment with its
own time and reason.

## 1. Question

TF-1 left 11 PDMS-zero rows under Tc20s-tracked (10 DAC, 1 NC) on the 1,018 held-out navtrain targets. 6 of the 10 DAC
rows fail even tracker-transparent, so they belong to the policy's path; about 4 are tracker residue. Two zero-training
label levers attack these rows:

* **L2b, margin shaping (PRIMARY).** Push the teacher's own 20-pose path laterally away from the drivable-area edge until
  every footprint corner keeps m* = 0.60 m, then derive the P5-S Tc20s label from the shaped path.
* **L3, DAC-certified fallback (SECONDARY).** Use the first label in [Tc20s, T, M] whose LQR-tracked footprint passes
  NAVSIM's DAC rule. If none passes, use the one with the largest tracked margin.

Admissibility. Both levers read only the static map (the scorer's own drivable polygons, queried around the t0 ego), the
t0 ego state and NAVSIM's public LQR tracker plus bicycle model. Neither reads the log's future: no future agents, and no
NC, TTC or EP verdicts. L3 reads only the `dac` field of the tracked arms, and NAVSIM's DAC is a function of the states
and the map only. This is privileged map geometry at t0, which is admissible for teacher LABELS (PI, 2026-08-03). Neither
lever is something the paper's teacher does, so a gain here is a label-quality gain. It is not evidence about checkpoint
identity.

## 2. Population, tier, estimator

* Population: the same 1,018 held-out rank-0 navtrain targets (11 logs) as D1, P5, P5-S, E2 and TF-1. Not navtest.
  The K ledger is unchanged.
* Tier: NAVSIM v1.1 PDMS, non-reactive pseudo-simulation, vendored navsim @ 3e8291b. One PDMScorer call per (log, token)
  with all arms in the same call, as in TF-1.
* Estimator: log-cluster bootstrap over the 11 logs, 10,000 draws, seed 20260927, 95 % percentile interval, ratio mean
  (TF-1's code verbatim). It answers *another draw of LOGS*. Labels, tracker and scorer are deterministic.

## 3. Arms (one scorer call per row)

| arm | definition |
|---|---|
| T_tracked | to_navsim(t20), LQR. Identity control: must equal TF-1's T_tracked |
| Tc20s_tracked | the banked P5-S cubic label, LQR. Identity control: must equal TF-1. **Baseline** |
| Tc20s_re_tracked | Tc20s re-derived here (P5's `iterate20`, cubic) from the banked t20. Derivation identity |
| **M_tracked** | margin-shaped path (section 4), then `iterate20` cubic, LQR. **PRIMARY** |
| Mw_tracked | the same shift profile with its sign flipped (toward the edge), then `iterate20`. **Deliberate regression** |
| CERT_tracked | L3 selection over [Tc20s, T, M] using only their tracked `dac`. **SECONDARY** |
| H_tracked | the logged human, LQR. Identity control |
| T_transparent | t20 as executed (E2's builder). Identity control |
| M_transparent | the shaped path as executed. Descriptive: does shaping remove the policy's own off-road? |
| T_sub20_tracked | t20 submitted as 20 poses at 0.2 s (`TrajectorySampling(20, 0.2)`), LQR. **Protocol probe P-S20** (section 6) |

## 4. Margin shaping, frozen

* Polygons: the scorer's `drivable_area_map` geometries of the types the scorer's DAC reads (ROADBLOCK, INTERSECTION,
  DRIVABLE_AREA, CARPARK_AREA), unioned. Footprint: Pacifica, `navsim_dac.corners_global` on the 20 label poses.
* For each pose i = 1..20, compute these quantities:
  * dmin_i is the minimum signed corner distance to the union boundary (positive inside).
  * The deficit is δ_i = max(0, 0.60 − dmin_i).
  * g_i is the unit direction from the worst corner toward the interior.
  * n_i is the pose's left normal.
  * The shift is sign(g_i · n_i) · δ_i / max(|g_i · n_i|, 0.5), capped at 1.0 m.
* Rate limit and anchor. The positive and negative parts are each dilated by a triangle of slope 0.10 m per 0.2 s step.
  They are then capped at 0.10 · i, because the origin at t = 0 cannot move. This allows at most 0.5 m/s of lateral
  rate.
* Repetition. The steps above are run 3 times, each on the shaped path, adding the increments. The final cap is
  |a_i| ≤ 1.0 m.
* Applying the shift. x, y move by a_i · n_i. The heading moves by atan2(a_{i+1} − a_{i−1}, s_{i+1} − s_{i−1}), on the
  original arc length s, one-sided at i = 20. It is unchanged where that arc span is under 0.2 m.
* A path whose every corner keeps ≥ 0.60 m gets no shift, and its M label must be bit-identical to Tc20s_re (control C3).
* m* = 0.60 m is D3's human median class-C margin, the same constant as P5's `MARGIN_M`. It is fixed and not swept.

## 5. Controls (all must pass, or no verdict is read)

* C1: T_tracked, Tc20s_tracked, H_tracked and T_transparent must equal TF-1's per-row values bit for bit (max abs diff
  0.0 on every sub-score).
* C2: Tc20s_re must equal the banked Tc20s label within 1e-9 m on every row, and Tc20s_re_tracked must equal
  Tc20s_tracked.
* C3: on rows with zero shift, the M label must equal Tc20s_re bit for bit.
* C4: deliberate regression. Mw must have ≥ 10 more DAC-zero rows than Tc20s_tracked.
* C5: liveness. M must differ from Tc20s_re on ≥ 100 rows.
* C6: coverage 1,018 / 1,018, 0 errors.

## 6. Outcomes, committed in advance

**PRIMARY: M − Tc20s, tracked, paired.**

* **A, L2b WORKS.** All four of these hold:
  * M has ≤ 4 DAC-zero rows (Tc20s has 10);
  * the ΔPDMS 95 % CI lower bound is > 0;
  * M's NC-zero rows ≤ Tc20s's + 1;
  * EP on rows alive in both arms is ≥ Tc20s's − 0.5 points.

  ⇒ M becomes arm 2 of PREREG_TF_N1 (navtest).
* **B, PARTIAL.** The ΔPDMS CI lower bound is > 0 but DAC-zero > 4, or DAC-zero ≤ 4 but the CI includes 0.
  ⇒ name the rows left; L2a (privileged selection among the teacher's TTS samples) goes next on them.
* **C, REFUTED IN THIS FORM.** Any one of these: the ΔPDMS point is < 0, NC-zero rises by ≥ 2, or the EP-alive loss is
  > 0.5. ⇒ L2a goes next.

**SECONDARY: CERT − Tc20s, tracked, paired.**

* **WORKS** if both hold: DAC-zero ≤ 4, and the ΔPDMS CI lower bound is > 0.
* Otherwise **NOT SHOWN**.

**PROTOCOL PROBE P-S20: T_sub20 vs T_tracked and T_transparent.** It is descriptive for the levers and decisive only for
the protocol question.

* **S20-CONSISTENT** if both hold: T_sub20 DAC ≥ 99.0, and T_sub20 PDMS ≥ T_transparent − 1.0. Reading: a standard NAVSIM
  submission at the teacher's native 0.2 s sampling reproduces the transparent level under the LQR. So the paper's 95.8
  can be LQR-tracked with no pre-distortion.
* **S20-NO** if T_sub20 DAC ≤ T_tracked DAC + 2.0. Reading: the 8-pose interface is not the cause.
* **S20-PARTIAL** otherwise.

## 7. Cost

CPU only, ≤ 3 processes, `CUDA_VISIBLE_DEVICES=-1`. ESTIMATED ≤ 1 CPU-h (TF-1: about 200 rows per shard in about 3 min,
plus 3 × `iterate20` per row).
