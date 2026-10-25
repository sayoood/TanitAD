# refcv9 L2 -- HANDOFF (cruise false-braking; lon-head calibration)

Everything lives in `C:/Users/Admin/refcv8_pf/refcv9_l2/`. Nothing is committed or staged (brief: do NOT git add/commit/push). Nothing was written to G:. Thor untouched.

## State at hand-off
* PREREG_L2.md sealed (sha256 4267cd684f38a0163fe1d5bf7c96a8b3ec884d48a01a3474d473b557c3bfb06e) + PREREG_L2_AMENDMENT1.md sealed (c463ba5556ab4527c8331c26346a1e35db5e22fc8eb2579132e7a4c4244ff692). DEVIATIONS.md lists D1-D4.
* 0-GPU result banked: `result/zero_gpu_nocal/ZERO_GPU.json` (K1 stage bit-equal, K2 populations + register literals all PASS). Exploratory post-seal: `result/Z4_nav_split_EXPLORATORY.json`, `result/Z5_label_split_EXPLORATORY.json`.
* ONE GPU chain `code/chain_l2.sh` queues on the shared gpu_lock behind L1 (and possibly P4). It needs L1's EMA-48,852 TRAIN bank (`refcv9_l1/bank/ztac_train_s0.npz`, 4,855 windows) to fit the calibrator; L1 produces it after its two eval rolls. Without it CAL is PENDING (NOTRAINBANK), CONST/ORACLE/NATIVE/IDENT still run.

## Check
```
tail -f C:/Users/Admin/refcv8_pf/refcv9_l2/logs/chain.log        # ZZL2<NAME>ZZ markers
```
Markers in order: START, IDENT, PREP, [COMMITWAIT], LOCKED, TRAINBANK, ROLL_native300_s0, ROLL_ident300_s0, ROLL_const_cruise_s0, FIT, ZEROGPU, GATE after_const, ROLL_cal_s0, GATE after_cal0, ROLL_oracle_s0, [ROLL_cal_s1], GPUFREE, SCORED, END. Failures: ABORT, NOLOCK, STOP, SCOREFAIL.
Assert on the ARTIFACT, not an exit code: each ROLL marker carries `windows=<n> want=<n>`; the final record is `bank/trace_<tag>.npz` (meta_json.final true).

## Read the result
`result/final/VERDICT.md` (mechanical: PASS / PASS-AT-SHIPPED-BASELINE / FAIL / PENDING per PREREG section 6), `L2_RESULT.json` (all cells, CIs, controls K1-K8, outcome O1-O6), `L2_RESULT.md`. `result/final` does not exist until `ZZL2SCOREDZZ`. Interim gates: `result/gate_after_const`, `result/gate_after_cal0`.

## Do NOT
* edit any file in `code/` while the chain runs (bash reads scripts lazily);
* kill by name; kill only the PID in `logs/chain.log` ZZL2STARTZZ (and its child), then `gpu_lock.py release --job refcv9-l2-lonhead --pid <that pid>` if the trap did not;
* touch `refcv9_l1/` or the lock when it is L1's.

## Expected wall clock (ESTIMATED from L1's logged eval roll: 139 episodes / 4,754 windows in ~3,600 s per seed)
4 full rolls (CONST s0, CAL s0, ORACLE s0, CAL s1) ~4.1 h + two 300-window smokes + loads ~0.3 h = ~4.4 h on the GPU (< 6 h, so both inference seeds for CAL are kept; CAL s1 is skipped only by the pre-registered gate).

## If the verdict is FAIL / O4
The next arm is the label-timing arm (PREREG_L2.md section 9). Pre-launch, cheaper and 0-GPU-first: see the report's "next lever".

## Next lever (written BEFORE the GPU outcome, from the 0-GPU readings; Rule Zero item 4)
MEASURED 0-GPU (compliant GT-cruise n=1,856, 254 false decels at seed 0; `result/zero_gpu_nocal`, `Z4/Z5/Z6_*_EXPLORATORY.json`; Z4-Z6 are post-seal descriptive, never bars):
* Even if the head said CRUISE everywhere, FD would stay ~0.131 (Z3, ESTIMATED association): head calibration cannot reach 0.10. 94 % of false decels are FOLLOW-nav windows; where the GT tactical label says cruise/follow (n=1,282, 73 % of false decels) the head already says decel in only 8.6 % but the stage plan brakes in 14.5 %.
* The same split bounds the LABEL-TIMING arm: only the label-decel group (n=215, 10.6 % of false decels) is a timing-ambiguity population, so relabelling can reach at best FD ~0.12.
* The false decels are real braking, not threshold-edge noise: median plan dv -2.59 m/s (edge is -1.5), 25 % deeper than -3.8, 10 % deeper than -5.1. A pure end-speed shift (oracle-tuned, upper bound) trades 1:1 -- FD 0.10 costs MD 0.469 -> 0.54 -- so it fails the MD bar.
=> The cruise brake is a SELECTION/CANDIDATE event downstream of the lon class (fan of 32 candidates, per-candidate score, decoder profile). Cheapest experiment (one dev-box GPU roll, ~1,400 FB/CT windows, ~1 h): re-run the fan at the shipped EMA-48,852 with per-candidate speed profiles + scores banked (adapt `brake_locus_reroll_r8.py`, which was measured at 30k CLIP-fed nav and must be re-derived at per-frame nav) and read SELECTION (a cruise candidate existed and scored lower) vs CANDIDATE (no cruise candidate in the fan). SELECTION => a TRAIN-fitted selector/score correction (L1-type); CANDIDATE => decoder/fan-allocation lever. Needs: GPU lock slot (~1 h), no new data, no PI compute decision.

## L2-B (brake LOCUS: SELECTION vs CANDIDATE), written 2026-10-25 ~15:10
Files: `PREREG_L2B.md` (+`.sha256`), `DEVIATIONS_L2B.md`, `code/l2b_locus_roll.py`, `code/l2b_score.py`, `code/l2b_verdict.py`, `code/chain_l2b.sh`; tests `smoke/l2b_roll_capture_test.py` (pass), `smoke/l2b_dryrun.py` (planted shares read exactly; class-swap mutation detected). Heavy data on F: (`F:/refcv9_l2_locus/`).
Check: `tail -f logs/chain_l2b.log`. Markers: START, IDENT, [COMMITWAIT], LOCKED, ROLL_smoke_s0, SMOKEGATE, ROLL_s0, SEED1 (run/skipped), [ROLL_s1], GPUFREE, SCORED_s0/s1, VERDICT, END (failures: ABORT, NOLOCK, STOP, SEED1FAIL). Result: `result/l2b_final/VERDICT.md`, `result/l2b_s{0,1}/L2B_RESULT.{json,md}`.
Do NOT edit `code/` while it runs; kill only the PID in ZZL2BSTARTZZ. It queues on the same gpu_lock behind P4 (job refcv8-p4-gtmap).
