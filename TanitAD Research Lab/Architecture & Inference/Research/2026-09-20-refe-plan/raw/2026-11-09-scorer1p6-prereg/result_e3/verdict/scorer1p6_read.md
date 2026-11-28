# SCORER1P6 read

**Status: READ**

- Tier: E3 = T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop); E2 = selection tier: PDM truth of P6's own 64 proposals on E2, the arm judged by which proposal it picks.
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs').
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check).
- Card: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_SCORER1P6.md` sha256 `adb5aafd4fb8abdfd1a53d9cc5ff865a89bd20f9008c701cf5d308dac595abed`

## Gates

- G0-PICK: **PASS** n_equal_to_banked_pick=12146, n=12146
- G0-P6CSV: **PASS** n_tokens_absent=0, n_invalid_rows=0
- G0-E2: **PASS** n_e2=2000, n_missing_in_table=0, n_missing_in_world=0, n_pdms_bit_equal=2000, n_sub_bit_equal=2000, max_abs_pdms_diff=0.0, formula_max_abs_diff=0.0, all_finite_valid=True, pick_is_argmax_of_p6_logits=2000, RED_shifted_pick_n_differ=1402
- G0-E3[C0]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1E]: **PASS** n=12146, n_tokens_not_bit_identical=0

## Verdict (written before ENS3)

- Adopted arm: **None**; verdict: **NO ADOPTION**
- Goal status (adopted): None; per arm: {'C0': 'BELOW GOAL', 'C0r': 'BELOW GOAL', 'S1E': 'BELOW GOAL'}
- Outcome: **O5 SWAMPED** (all matches: O5, O4)
- Next lever: NOT PROVEN: S1E seed 1 and a second C0 replicate
- F (PDMS, E3) = +0.3796; F (M1) = +0.0209; dRho_C0 = -0.1279

## Endpoints

| arm | PDMS E3 | vs P6 | changed picks | goal |
|---|---|---|---|---|
| P6 | 91.196 [90.410, 91.984] | - | 0 | |
| C0 | 91.229 [90.413, 92.043] | +0.0325 [-0.1584, +0.2307] | 5998 | BELOW GOAL |
| C0r | 91.608 [90.827, 92.396] | +0.4120 [+0.1971, +0.6364] | 5852 | BELOW GOAL |
| S1E | 91.271 [90.461, 92.098] | +0.0744 [-0.1273, +0.2720] | 5517 | BELOW GOAL |

| arm | M1 dRho level | M1 S1E-C0 style diff | G1 TTC/NC/DAC pts | G2 d-regret | G3 overspeed diff | guards |
|---|---|---|---|---|---|---|
| C0 | -0.1279 [-0.1823, -0.0722] | n/a | +0.173/+0.004/+0.148 | +0.0593 [-0.4528, +0.5723] | -0.0194 [-0.0464, +0.0110] | False |
| C0r | -0.1070 [-0.1643, -0.0478] | n/a | +0.527/+0.123/+0.181 | -0.5054 [-0.9960, -0.0064] | +0.0116 [-0.0179, +0.0428] | True |
| S1E | -0.0961 [-0.1495, -0.0421] | +0.0318 [+0.0237, +0.0403] | +0.000/-0.045/+0.025 | -0.0646 [-0.5199, +0.3899] | +0.0039 [-0.0205, +0.0303] | True |

M1 n per head (CRUISE tokens entering): {'TTC': 119, 'NC': 86, 'DAC': 178}

## Family clause (arm - P6, paired CIs)

- C0: adverse = ['EP']
- C0r: adverse = none
- S1E: adverse = none

- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.

## ENS3 (read after the verdict; cannot change it)

- NOT READ: harness scores missing for 326 ENS3 changed picks (run emit-changed --arm ENS3, score, pass --ens3-csv)
- none: ENS3 has no replicate arm, so no 2F rule can be applied to it (stated per card section 5)

## Ambiguities applied

- A1 plan speed v4: the arms' outputs carry the executed plan as 8 NAVSIM poses at 0.5 s, not the 20-step 0.2 s trajectory the C3 read used. v4 is therefore the chord of the LAST step (|p[7]-p[6]|/0.5) for plans and for the human future alike (same estimator on both sides). Overspeed (v4/v0 > 1.15), slow share (< 0.85) and the Spearman ratio v4/v0 use it. Not tuned: it is the only plan representation in the outputs.
- A2 CRUISE: selc6_analyze.cruise_formula applied to the export's own fields on the 0.5 s grid: human future finite, v0 = |ego_velocity| > 5, |ego_acceleration_x| < 0.3, |v4_gt - v0| < 1.0 (v4_gt the human last-step chord). E2/E3 tokens have no 20-step ego-probe record, so tokens_cruise.json cannot be used there.
- A3 M1 token set per head: a token enters head h only if its TRUE labels vary AND every arm's predicted head is defined on it (a common set, so every arm difference is paired); n per head is reported. Mean over TTC, NC, DAC of the per-head means.
- A4 M1 bar read as written: S1E - C0 >= +0.5 * |dRho_C0| (dRho_C0 = C0's own mean-over-heads gap), CI-lo of S1E - C0 > 0, and |S1E - C0| > 2 * |dRho_C0r - dRho_C0|. The M1 floor is the same C0r-C0 pair as P1's.
- A5 F is |C0r - C0| on the point estimates (PDMS x100 on E3 for P1). If C0r is absent F is undefined and no bar that needs F is cleared (conservative).
- A6 'clears P1 against P6' for S1E means its own P1 (vs C0 with the 2F rule AND vs P6 CI-lo > 0). For O4 'no arm clears P1 against P6' is read as: neither C0 nor S1E has CI-lo > 0 and |diff| > 2F against P6.
- A7 Outcome overlap: O1..O6 are not mutually exclusive as written. Every match is listed (matches); the primary label follows the fixed order O1, O2, O3, O5, O6, O4, Unnamed (most specific first; O5 NOT PROVEN before O4/O6). O3 (S1E harm) is evaluated whether or not C0 is adopted.
- A8 Family clause covers LONGITUDINAL (EP, TTC, target-speed accuracy, speed MAE, along-track MAE and final error, accel MAE, overspeed share) and LATERAL (DAC, heading, curvature, yaw-rate, cross-track MAE and final error) components, adverse iff the paired CI of (arm - P6) lies entirely on the worse side. No multiplicity correction is applied (card silent). NC, C, DDC are reported with CIs but sit in G1/PDMS, not in the clause. TACTICAL components are reported with paired CIs and do not enter the clause (the clause text names longitudinal and lateral).
- A9 Per-token family components are masked means over the horizon steps (the four_families masks); a token enters a component's bootstrap only where the component is defined for both arm and P6. Distance-keeping is UNAVAILABLE at this tier (card).
- A10 E2 and E3 use the arm's logits with the planner's navsim_v1 rule WITHOUT the lane factor (6 outputs): pick = argmax of p_NC*p_DAC*(5 p_EP + 5 p_TTC + 2 p_C)/12 in torch float32, first index on ties. All 64 proposals are treated as valid (E2 table validity is gated).
- A11 Unchanged tokens carry P6's banked csv value; a changed token's value is looked up as the score of the SAME proposal index, from P6's csv, the E2 table, or any arm's harness csv (proposals are bit-identical across arms by G0-E3, so a (token, k) score is arm-independent; conflicts stop the read).
- A12 Boundary comparisons use a 1e-9 slack in the arm's disfavour only where the card bar is a decimal (G1 -0.5 points, G2/G3 strict '<', goal 94.55): see 'bar' helpers; CI-lo > 0 and |diff| > 2F are strict with no slack.

## Not implemented / not verified

- E1 (C3, 1,058 tokens) is reported as a stratum-weighted point only when --c3-tokens is given; no interval (descriptive).
- Closed-loop read (card section 6) and the S1 / GPU arms are outside this tool.
- ENS3 has no replicate floor; stated with its result.
