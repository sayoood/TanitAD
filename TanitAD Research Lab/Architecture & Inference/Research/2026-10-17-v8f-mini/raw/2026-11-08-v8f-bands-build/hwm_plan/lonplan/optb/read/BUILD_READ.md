# OPTB read side: build report (2026-11-18). Tier T0. Staged only; nothing committed or pushed; box 2, the pod and git untouched.

## Headline
* **C10' on R35: PASS at both chassis (MEASURED).** s_swap, acceleration, 141 EVAL clips, IA-3's folds (2, `lr_lib.fold_of`), 12,827 anchors:
  R35_s0 0.2280 [0.2020, 0.2520]; R35_s1 0.2223 [0.1952, 0.2473]. Estimator: episode-cluster bootstrap, B 2,000, seed 0 = another draw of episodes, R35's weights and the probe FORM held fixed
  (not another training run). Artifacts: `c10p/RESULT_C10P.{json,md}`, `step0/ia3pk_R35_s{0,1}.json`. Train-clip Amendment 1 values were 0.201 / 0.203.
* Step 0 reference (card predicted G_aa about 1.2): through the frozen train-fit RA the cycle gain on EVAL clips is 1.255 [1.134, 1.374] (s0) / 1.246 [1.125, 1.366] (s1);
  IA-3P-K (cross-fit RA, s_swap) reads 0.990 [0.876, 1.129] / 0.891 [0.785, 1.014]. The two differ because the cross-fit probe is a different (eval-fit) ridge with its own slope; the frozen-probe number is the
  term's own cycle gain (TARGET-READ), IA-3P-K is the MECH-MOVED instrument. Both are step-0 references for the arm comparison, not bars.
* **MM to decide / wire:** (1) land `read/` with the tree; (2) the planner stage that makes `plans_hwm_oracle_s{0,1}.npz` per arm is NOT part of this package (35.6/27.3 min HWM per seed, ~2 h per checkpoint,
  from AMEND_HWMPLAN_3 timings) and needs a chain; (3) ruling on HARM-vs-SEEDS-DISAGREE precedence (below); (4) `controls.c2_c3_hold` and `band_met_lost` are manual inputs (nothing computes BAND-MET here).

## Files and hops
* `ia3pk_read.py`: `build_pairs` (:90, anchors with all K+1 frames ok and finite actions; z1 via `wm.z1_from_mean`, actions via `wm.scale_actions`), `crossfit_ra` (:112, ridge as `amend1_kstep.py`, folds = `fold_of`,
  lambda on inner clip-grouped folds), `contrast_read` (:136, two +/- delta contrasts on channel 0, one batched `stack.rollout(1, z0, a, context=1)`), `s_swap_from_moments` (:162), `read_core` (:176),
  `read_checkpoint` (:243: asserts adapter sha equals the cached frames' and `wm.proj` equals R35's, else REFUSED), `mech_moved` (:278).
  CLI: `ia3pk_read.py read --ckpt <dir|ckpt_last.pt> --chassis k --tag <tag> --out <json>`; `ia3pk_read.py mech --dir <dir of ia3pk_<tag>.json> --out <json>`.
* `optb_decide.py`: `better_worse`, `chassis_label`, `overall_label` (pure logic), `setup`/`Scorer`/`run` (imports the existing `hwm_eval` reader from `lonplan/build/reader_src` and `run_hwm.load_ws`; nothing copied or edited).
  CLI: `optb_decide.py --manifest M.json --out OUT.json`. Manifest format in its docstring.
* Tests: `test_ia3pk.py` (14), `test_optb_decide.py` (6): 20 passed (CPU, 2.9 s). Deliberate-regression arms that go RED: no-response world reads gain exactly 0 with CI containing 0; wrong-channel world reads 0;
  no action signal in the real transitions makes C10' read FAIL; shuffled actions make C10' FAIL; MECH-MOVED goes RED on a pair gap below F_T,k, overlapping CIs, a missing arm (NOT-COMPUTABLE); a planted gain of
  0.5 / 1 / 2 is recovered. Reproduction: the stand-in pair (A1_s0 vs A0_s0, B 2,000) reproduces the banked `lonplan/readout/c_A1_A0_s0.json` acc_mae and yaw_mae 0-2 s delta, lo, hi to the banked 4-decimal rounding.

## Design choices (each is a reading of the card, flag if wrong)
* IA-3P-K contrast size = the term's 1 sd, 0.4203 WM units (`--delta`), not IA-3's 1.0 m/s^2. Anchors = every valid start (stride 1), not the term's 9 per clip.
* The real pairs (z_t, z_{t+10}) are arm-independent (adapter and `wm.proj` frozen), so RA, s_swap and F_T,k share one calibration per chassis; the arm differs only through its rollout. Reader refuses if `wm.proj` moved.
* s_swap and the gain are recomputed from the same bootstrap resample. MECH-MOVED uses disjoint per-arm CIs plus gap > F_T,k exactly as written; the paired per-clip moments are saved (`*_moments.npz`) if the MM wants a paired difference.
* CEILING: some deciding quantity q with HWM-ORACLE(arm) - HOLD CI wholly below 0 for both lever arms of the chassis, other q not WORSE; overall needs both chassis to agree. HARM at either chassis is HARM (card: "WORSE at either
  chassis"); that outranks SEEDS-DISAGREE: an MM ruling is needed. Within-chassis sign split on a deciding quantity = SEEDS-DISAGREE. F_I = max |seed1 - seed0| over every arm and baseline with both seeds (global, per card).
* TACTICAL goal/anchor selection: NOT COMPUTABLE (planner has no decision output). STRATEGIC: full 6 s band, structural n/a items from the reader, never a pass.

## UNVERIFIED / not built
* No FT checkpoint exists, so `read`/`mech`/`decide` were never run on real arms; the decision smoke (`smoke/`) uses stand-ins (A1/R/A0/S plans of LON-PLAN), chassis 0 only; its labels are meaningless by design.
* BAND-MET and C2/C3 are not computed here. The planner stage and the HF phase-1 R35 plans are not run. The 3-FT-NULL F_I over seeds 2-5 uses whichever arms have a seed-1 plan.
* Cost per checkpoint: IA-3P-K read MEASURED 5.2-5.8 s wall CPU (2 threads) + model load; decide scoring MEASURED 9 s at B 200 (about 2 min at B 2,000) per smoke; the planner (HWM, 2 seeds) is the real cost, ESTIMATED ~2 h per
  checkpoint (INHERITED from AMEND_HWMPLAN_3 timings, not re-measured). The card's 2.2 h is dominated by it.

## Manifest (all on F: scratch only unless stated; NONE banked to git/HF: single copy each)
| artifact | path (under F:/TanitAD-scratch/actrsp/optb/read/) |
|---|---|
| C10' result | c10p/RESULT_C10P.json, .md, make_result_c10p.py |
| R35 step-0 reads | step0/ia3pk_R35_s{0,1}.json (+ _moments.npz), run.log, run_step0.sh |
| IA-3P-K reader + MECH-MOVED | ia3pk_read.py, test_ia3pk.py |
| deciding-read driver | optb_decide.py, test_optb_decide.py |
| decide smoke | smoke/manifest_smoke*.json, decide_smoke.json, decide_smoke_B2000.json, run*.log; run_smoke.sh, run_smoke2000.sh |

## Deletions log
Deleted my own temp: tmp/pt1, tmp/pt2, tmp/pt3 (pytest basetemp, < 1 MB each), tmp/smoke_s0.json + smoke_s0_moments.npz (smoke_frac run, < 20 KB), __pycache__/.pytest_cache, empty tmp dir. Re-creatable; nothing banked was removed.
