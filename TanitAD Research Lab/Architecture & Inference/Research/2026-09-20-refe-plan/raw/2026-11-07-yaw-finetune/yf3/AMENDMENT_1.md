# AMENDMENT_1 to PREREG_YAWFT (stream YF3, written 2026-11-07 BEFORE any trained arm was evaluated)

`PREREG_YAWFT.md` is untouched (sha256 `d28a630fa650a8fe50488dea82f01e1d61a5798347d7f522926b2b8215145b74`, verified at YF3 start). No threshold, bar or decision text in it is changed by this file.
This amendment does two things only: (A) states how the under-specified mechanics were operationalised, so the analysis is mechanical; (B) names the two places where the executable pipeline
differs in ROUTE from the PREREG's wording. Nothing below was chosen after seeing a trained arm's eval (no trained arm has been evaluated; the only evaluated checkpoint at writing time is the banked A0).

## A. Operationalisation (gaps in the PREREG text, filled before the data)
1. **Primary statistic.** Curvature MAE (1/m) of the executed pick vs the human future is a RATIO of per-window sums (sum of |kappa_pred - kappa_gt| over valid step pairs / number of valid pairs), the same pooled
   quantity families6's `lateral()` prints. MEASURED reproduction on the banked A0 seam: 0.018820 over 5,760 step pairs (banked `sft4_B_u400_families.json`: 0.01882, n 5,760); heading 1.9235 deg, yaw-rate 2.6396 deg/s,
   cross 0.2761 m all equal the banked block (`test_yf3_analyze.py`, 15/15).
2. **Pairing and pooling over seeds.** The paired difference is taken per seed, B_s - Ref_s, on the SAME log resample. The decision CI is the percentile CI of the SEED-AVERAGED contrast
   (mean over the two seeds of B_s - Ref_s) under the 10,000-resample, seed-20260927 log-cluster bootstrap (the resample indices are drawn exactly as `proxy_eval.boot` draws them; the estimator reproduces the
   banked SFT-4 PDMS contrast mean +3.041874 and CI [1.209549, 4.865133] to 1e-9). The per-seed CIs and the per-seed Deltas are reported beside it. "Both seeds agree in sign" = Delta(s0) and Delta(s1) are non-zero
   and of the same sign. F = |Delta(s0) - Delta(s1)|. "Mean Delta" in `F < |mean Delta| / 2` is the seed-averaged point estimate.
3. **Reference arm Ref.** R1: Ref = A0 for both seeds (valid because gate K3 proves arm C's checkpoint state_dict is `torch.equal` to u400 and carries no `ego_yaw` meta; the PREREG says C "is evaluated only if a
   gate fails"). If K3 is red, `build_yaw_ckpt.py` REFUSES to compose the C checkpoint (rc 3) and the pipeline stops FAILED (an instrument fault needs a human; there is no C checkpoint to evaluate). R2 / R3: Ref_s = C_s of the same rung and seed (the PREREG's primary contrast B - C).
4. **Relative improvement** = -(mean Delta) / (mean over Ref arms of their measured curvature MAE). At R1 the Ref is A0, whose measured MAE is gate-checked equal to the banked 0.0188 (G6), so the PREREG's
   parenthetical `>= 0.00094 1/m on A0's 0.0188` is the same bar; it is evaluated and reported as a separate line and is NOT used to relax or tighten the 5 % bar.
5. **PDMS non-inferiority** = the lower bound of the percentile CI of the seed-averaged PDMS Delta (x100 scale) >= -1.0. Per-seed PDMS Deltas, F on PDMS, and the six sub-scores are reported (guards).
6. **`families_adverse`** = `proxy_eval.fam_adverse(Ref families6 block, B families6 block)` per seed (families6's own episode-cluster intervals, unpaired interval non-overlap: the conservative reading
   `stage2_sft.py` used for SFT-4). ADOPT requires it empty for BOTH seeds; ONE adverse component in either seed makes the verdict REFUTED (the PREREG's "a lateral / longitudinal component separates adversely").
7. **Order of the rule.** VOID (a red gate) -> REFUTED -> ADOPT -> NOT PROVEN. ADOPT and REFUTED are mutually exclusive by construction (ADOPT needs primary upper < 0, PDMS lower >= -1.0, no adverse).
8. **TACTICAL** is reported as "n/a as a test: REFe has no tactical layer; families6 derives decisions from executed geometry" with the descriptive accuracy / kappa beside each arm. **STRATEGIC: PAUSED** (PI
   ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25): not a pass, not omitted. **Tier:** NAVSIM v1.1 pseudo-simulation of an OPEN-LOOP plan against logged agents; not the action-closed-loop tier.
9. **Strata** TURN (|omega at t0| > 0.05 rad/s, from the NAVTEST table) vs STRAIGHT: descriptive curvature MAE per arm, no decision weight.
10. **G1** is YT's control (DB vx / vy / ax / ay equal the bank's ego; every token present). YF3 cannot re-run YT's DB control; it asserts the part it can: every eval token is in the pinned NAVTEST table, every TRAIN / VAL
    token in the pinned NAVTRAIN table (G9), and both tables carry the pinned sha256. Marked as such in the analysis.
11. **G8 eval determinism** = arm R1 B_s0 re-run through the seam on every 10th sorted token of the 923 (93 tokens); plans must be `np.array_equal` to the main run's rows AND the shim must feed identical ego[4].
    **G8 CPU** = R1 B_s0 retrained once (same argv, seed, threads: OMP/MKL 12) and `col4` `torch.equal`. G8 is measured on R1 and CARRIED to R2 / R3 (UNVERIFIED there: the R2 / R3 optimiser touches more
    parameters; their CPU determinism is not separately re-run).
12. **Rung chaining (Rule Zero).** After a rung's verdict: NOT PROVEN -> next rung; REFUTED -> next rung ONLY IF every gate was green (the PREREG: "continue ... only if the harm is not an instrument fault";
    G7 = the shim frame, G2 = the sign, both gates are green by construction of this rule; the lr leg of the diagnosis is the VAL curve in `metrics.jsonl`, printed in the analysis); VOID -> STOP, FAILED marker
    (an instrument fault needs a human). After R3 NOT PROVEN or REFUTED the PREREG's elimination text and next levers are written to `FINAL.md`; ADOPT stops the chain at that rung
    and names (a) the 12,146-token confirmatory read, (b) the closed-loop A/B, (c) P6 final as the NEXT steps (not auto-launched: 4.5 h of GPU on a decision that is the reader's).

## B. Route differences from the PREREG's wording (not threshold changes)
1. **`stage2_sft.py` cannot carry the shim.** It launches the seam as a subprocess, so a launcher around it never reaches the planner. `yf3_eval_arm.py` re-implements its steps (seam, W3 harness via `score_e.py`, families6)
   with the SAME arguments (`--rule navsim_v1_lane`, same frames, DB, export, harness pins verified first) and wraps ONLY the seam in `yf3_launch_g7.py`, which delegates to YT's `ego_yaw.main` UNCHANGED and additionally
   RECORDS the ego[4] it fed per token (the return value is untouched) so G7 is read from the live seam. Every arm, A0 and C included, goes through the same launcher; YT's shim stays out of the planner for a checkpoint
   without meta `ego_yaw` (checked: `shim_installed False`, 0 calls). G6 (A0 re-run == banked plans and scores, bit for bit) therefore also certifies the launcher path.
2. **Harness side effect.** `score_e.py` writes its run folder under `<navsim root>/exp/w3_navtest_v1/runs/<label>` (outside YF3); labels are `refe_yf3_<tag>_<arm>`. Unavoidable; named in the manifest.
3. **`build_yaw_ckpt.py` gate K4 was mis-specified for arm C (found by the YF3 smoke, before any real trained arm existed).** MEASURED on the smoke arm R1_C_s0: `K1 True, K2 True, K3 True, K4_col4_equal_final False -> ZZYAWCKPT_GATE_RED`. For arm C the leaf `col4` is the
   zero-init that is never deployed (yaw 0 at train AND eval), while the composed checkpoint correctly keeps u400's own column 4, so "composed column 4 == final col4" can never hold for C: the C checkpoint (K3's
   C == A0 identity, and every R2 / R3 C arm) was impossible to compose. YF2's file is pinned and untouched; YF3 uses `build_yaw_ckpt_yf3.py` (diff: `build_yaw_ckpt_yf3.diff`, 1 conditional + the path of
   `yaw_common`): K4 for arm B is unchanged (`col4_equal_final`); K4 for arm C becomes "the composed column 4 equals u400's own column 4" (`K4_col4_untouched_for_C`). K1, K2, K3 and the compose function are byte-identical to YF2's.
   Nothing about a threshold or a decision changes; this is the gate that guards the C == A0 identity, now actually reachable.
4. **Adapter frame guard at small n.** `scenes_to_win(verify=True)` runs a statistical lateral-sign check ("90 % of the turning windows agree"); it fires on a smoke subset of ~18 tokens (7 of 8 windows) and is
   not decision-grade there. `yf3_perwindow.py` keeps it ON for any set of >= 100 tokens (the real 923 run, and the 110-token smoke) and skips it, saying so, only below 100. families6 itself is unchanged.
