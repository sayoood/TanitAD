# PREREG_V8F_MINI — the PI's three-layer v8F on frozen DINOv3-L features (track M), and data vs epochs

**Committed:** 2026-10-17 (Europe/Berlin), BEFORE any data. It is governed by `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §13 (the PI rulings of 2026-10-17) and extends `PREREG_HJ_STAGE0.md`. RULE ZERO applies: a missed bar is reported FAILED and the named next lever runs.

## Common settings

* **Grid:** the refav1 trunk-probe grid: 141 v7.2 EVAL episodes, **2,399 windows**, horizons 2 s and 6 s, T1-equivalent open loop. P10's banked numbers live on this grid (2 s 0.2229 / 6 s 2.9257 m).
* **Features:** the pooled frozen DINOv3 ViT-L/16 cache from S4 (4×10 primary, 8×20 secondary). TRAIN clips are disjoint from the eval clips by id **and by recording session**.
* **Estimator:** the paired episode-cluster bootstrap, B = 2,000. Every row names the variance it answers.
* **Seeds:** **training seeds 0 and 1** for every arm. The seed floor is F = \|arm_s0 − arm_s1\|.
* **Inputs, never labels:** t0 ego dynamics (the P10 features), max target speed (v8 sidecar), and nav command + distance (v9).
* **Metric families:**
  * the four families in full; **STRATEGIC is un-paused** (PI ruling of 2026-10-17);
  * plus **CONSISTENCY**;
  * plus the "acts" check: mean |a| and the zero-accel fraction.

## Arms

| arm | what |
|---|---|
| `M1` | S (1 Hz) → T (2 Hz) → O (10 Hz) on the scene features. O emits a 6 s residual on P10. Sample-and-hold between layers; scheduled sampling p_GT 1.0 → 0.2; consistency penalties on |
| `M1-navonly` | M1 with **no scene features** in S and T. This is the strategic and tactical "vision helps" control, and the PI's literal nav + distance-only strategic layer |
| `M1-navshuf` | M1 evaluated with the nav command + distance taken from another episode (fixed permutation, seed 0). This is the nav-use control |
| `M1-nocons` | M1 with the consistency penalties at 0. It isolates what they buy |
| `M2a` / `M2b` | M1 + an H-JEPA WM auxiliary (z¹ / z² / z³ on O / T / S cycles), **without / with IDM** (β 0.08, γ 100) |

## Hypotheses and bars

**H-MINI-K-1.** `best(M1, M2a, M2b) − P10`: ADE at 6 s < 0, separated, at both seeds, with \|Δ\| > 2F. Also read against `damp50` and `kd_x`. Event strata are reported: GT turn onset, stop, lead braking.
* PASS ⇒ AR-1, trajectory part.
* FAIL ⇒ plan §13.6, row 1.

**H-MINI-CONS-1.** CONSISTENCY(a) of the best arm ≤ P10's own rate on the same windows, separated; self-consistency (b) ≥ 0.90.

Violation rules, frozen now:
* **SPEED:** the planned max speed is more than 0.5 m/s above min(limit, SPEED_BAND v_hi).
* **STOP:** the planned speed is more than 1.0 m/s at the stop distance, or the plan passes the stop point by more than 2 m.
* **TURN:** the sign of the planned heading change differs from the goal side, while a TURN goal is active.
* **LEAD:** the minimum planned time gap is more than 0.2 s below the constraint.

`M1-nocons − M1` is reported; it is expected to be worse.

**H-MINI-STRAT-1.** Strategic checkpoint position error at 30 / 50 / 80 m.
* **Bar:**
  * M1 beats the **nav-only kinematic baseline**, separated at 50 m and 80 m;
  * `M1-navshuf` is **worse** than M1, separated (nav is used);
  * M1 beats `M1-navonly` (the scene adds strategic value).
* **The nav-only kinematic baseline:**
  * extrapolate the P10 6 s path to 30 / 50 / 80 m of arc;
  * when nav announces a TURN at distance d, go straight to d, then follow an R = 12 m arc to the nav side.
* **If M1 does not beat `M1-navonly`:** ship the nav-only strategic layer (the PI's literal spec), and record that the scene adds nothing at the strategic level at this scale.

**H-MINI-TAC-1.** v9 tactical goal **macro-F1** over the labelled cells, with IGNORE masked.
* **Bar:** M1 beats the majority class and `M1-navonly`, separated.
* **Reporting:** the per-class confusion matrix is reported. So is TURN timing error against the label's onset.

**H-DATA-SCALE-1** (plan D2). M1's recipe (or M2's, if it wins H-MINI-K-1) at three settings:
* **(a)** 25 / 50 / 100 % of the TRAIN clips, at matched optimiser steps S, where S = one epoch at 100 %;
* **(b)** 100 % at 2S steps;
* **(c)** 100 % at 4S steps.

Every setting is run at 2 seeds.

* **Two gains are compared:**
  * Δ_data = ADE₆(50 %, S) − ADE₆(100 %, S): doubling unique data at fixed compute;
  * Δ_epoch = ADE₆(100 %, S) − ADE₆(100 %, 2S): doubling compute at fixed data.
* **The outcomes:**

| result | meaning ⇒ next |
|---|---|
| Δ_data > Δ_epoch + 2F | **data is the lever** ⇒ D3 (the egomotion-mined extension) goes first |
| Δ_epoch > Δ_data + 2F | **epochs are the lever** ⇒ train longer on the 26 h first; D3 is deferred |
| both within 2F of 0 | neither binds at this scale ⇒ the architecture and the trunk (LoRA) are the lever |

* **Reporting:** the 25 % point and the 4S point are read for curvature and reported, but they do not decide.

## Amendment 1 (2026-10-17, before any M-arm data): the P10 reference is refit in the correct steer unit

* The banked refav1-grid P10 (2 s 0.2229 / 6 s 2.9257 m, "Common settings" above) was fit as a residual on a `kd_x` built with the steer angle integrated as a curvature (RETRACTION_LOG R46).
* On that grid the correctly-integrated echo `ha0_ext` reads 2.838 m, which already beats it.
* ⇒ **M1's residual base, and the P10 every H-MINI bar is read against, is P10 refit on the correct-unit `kd_x`** (`taniteval/taniteval/floors_kin.py`, `action_units="steer"`). Its 2 s / 6 s numbers are reported beside the old ones.
* **`ha0_ext` (correct unit) is added as a reported floor.**
* This makes every bar stricter, never looser. No arm has been trained.

## Amendment 2 (2026-10-17, evening, before any M-arm data): reference numbers, label release, and two rulings

Source: the M1-build package `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/` (`RESULT.md`, `raw/p10/p10_fit.json`, `LABEL_JOIN_REPORT.md`). No arm has been trained.

1. **The P10 reference, refit in the correct steer unit** (Amendment 1), on the 2,399-window grid:
   * 6 s ADE **2.5744 m** [2.327, 2.829];
   * 2 s ADE (first 10 knots) **0.2187 m**;
   * 6 s FDE 8.024 m.
   * The same script in the legacy unit gives 2.9281 m, against the banked 2.9257, so the pipeline reproduces.
   * The correct-unit `kd_x` reads 2.9396 m. S1 independently read 2.9396 for the same floor (`…/2026-10-17-p10-floors-refcv/raw/refav1_units_check.json`).
   * ⇒ **H-MINI-K-1 is read against 2.5744 m, not 2.9257.** The bar is harder by 0.35 m.
   * Artifacts: `raw/p10/p10_prior.pt` (the residual base) and `raw/p10/p10_eval_pred6.npy` (the comparison).
2. **Label release.** Train on the v9 **4,572-clip** release (md5 `83e30bd3…`), which labels 100 % of train windows. The f63ece41 release leaves 4.4 % unlabelled.
   * Eval: **139 of 141 episodes** carry v9 labels (2,365 of 2,399 windows).
   * TACTICAL, STRATEGIC and CONSISTENCY are read on those 139. The trajectory bar (K-1) is read on all 141.
   * The 2 missing episodes are a named gap, not a silent drop.
3. **CONSISTENCY floor, reported beside every reading.** The frozen SPEED rule flags **6.3 %** of windows on the human's own path, because the max-speed sidecar holds one limit per clip. The GT-plan control reads **7.9 %** overall. P10 reads **32.4 % [27.6, 36.9]**.
   * The rules and the bar are **unchanged**: the best arm must read ≤ P10's rate, separated, and self-consistency must be ≥ 0.90.
   * The GT control is the rule's own noise floor, so no arm is expected to go below it.
4. **TAC-1 ruling, made before any data:**
   * **primary = goal-token macro-F1** over the v9 tactical goal vocabulary, with IGNORE masked. This is the PI's "tactical goals" as a layer output;
   * **reported, not decisive:** lateral and longitudinal macro-F1, plus the confusion matrices.
5. **Strategic floor numbers.** The nav-only kinematic checkpoint baseline reads **2.01 / 4.82 / 11.15 m** at 30 / 50 / 80 m of arc.

## Amendment 3 (2026-10-17, late night, before any M-arm data): the three-layer v8F-mini gets the MEASURED YAW RATE, not the steer angle (PI ruling `D-PI-YAWRATE-1`)

* **The ruling (PI, verbatim):** *"the model should get measured ego motion and its history, instead of steering angle which represents a sort of action, the model should get the measured yaw rate"*. Same finding as `PREREG_HJ_STAGE0.md` Amendment 3: `actions[:, 0]` is `atan(2.9 · measured curvature)`, measured motion in an action's units, so it is **re-expressed, not deleted**.
* **Where the steer reached the model** (audit of `v8f_mini.py` and the data layer): the eight P10 features `p10f` (read by the ego token and by the prior), the `kd_x` `base`, and the WM auxiliary's action channel 1 (M2a / M2b only). No other input.
* **Binding:**
  * **P10-yaw-600 replaces P10-600** as the residual base and as the reference every H-MINI bar is read against. Features `(v0, a0, κ0)` with `κ0` from the measured yaw rate (`kin_mode="yaw"`, the `p10_pose` triple), `kd_x` integrated as a curvature. Recipe, clips (first 600 by sha12), stride 8 and seed 0 are unchanged.
  * The WM auxiliary's action channel 1 is the measured yaw rate (scale 0.4 rad/s; the steer scale was 0.3 rad).
  * The trainer and the evaluator default to `--kin-mode yaw`, need `--tracks native_tracks.pt`, and REFUSE a prior whose declared units are not the mode's. The floors are named `kd_x_yaw` / `damp50_yaw`.
* **Reference numbers, MEASURED** (eval-141, 2,399 windows, `raw/p10_yaw/p10_fit.json`, `…/2026-10-17-refav1-hj-mini/raw/p10_yaw_vs_steer.json`):
  * 6 s ADE **2.5817 m** [2.3285, 2.8410], against 2.5744 m for the steer P10-600 of Amendment 2;
  * 2 s ADE (first 10 knots) **0.2201 m**, against 0.2187 m;
  * 6 s FDE 8.035 m; the yaw `kd_x` reads 2.9444 m;
  * paired `yaw − steer`: **+0.0074 m [-0.0140, 0.0306]** at 6 s, **not separated** (episode-cluster bootstrap, B = 2,000: episode variance only).
  * ⇒ **H-MINI-K-1 is read against 2.5817 m**, not 2.5744. **This bar is 0.0073 m looser than Amendment 2's.** Amendment 1's "stricter, never looser" does not hold here; the loosening is the measured price of complying with the ruling, it is disclosed, and no M-arm exists to have been tuned against it.
* **No M-arm has been trained** under either representation (INHERITED from Amendment 2, not re-verified here); the steer representation remains selectable with `--kin-mode steer` for reproduction only and decides nothing.
* **Guards:** `stack/tests/test_yaw_rate_ruling.py` plus the parametrised end-to-end in `test_v8f_mini.py`; the deliberate-regression arms and the source mutations are the ones listed in `PREREG_HJ_STAGE0.md` Amendment 3.

## Amendment 4 (2026-10-18, before any M-arm data): the full corpus, the RA-1 holdout withheld, M1 vs M2b first (PI go)

* **The PI's go (verbatim, 2026-10-18):** *"Yes, queue v8F-mini M1 vs M2b on box 2 after RA-1 full finishes, on cache_full with a PREREG amendment, safe mode, stop on a crash."*
* **No M-arm has been trained under any amendment** (MEASURED 2026-10-18: no v8F-mini run directory exists on box 2, and none in the repo).

**1. Features.**
* The pooled cache is `D:\tanitad\cache_full` on box 2: 4,572 TRAIN clips + the same 141 eval clips.
* It was built by the RA-1 full build (`PREREG_RA1_FULL.md`, `H-RA1-FULL-1`), which passed a byte-IDENTICAL C0 rebuild control.
* This is the corpus the original handoff sized v8F-mini for (~310,642 windows). The 600-clip S4 cache is NOT used.
* `cache_full` also holds the 8×20 grid, so the first K-1 fail lever (plan §13.6 row 1) needs no new encode.

**2. TRAIN = 4,512 clips: the 4,572 minus clips 540:600 of the sha12 order.**
* Mechanism: the new trainer flag `--exclude 540:600`. These 60 clips are the RA-1 selection holdout (sha256 `f6a1dee0…`).
* Why:
  * **(a)** The residual base and the reference, P10-yaw-4512, was fit on exactly these 4,512 clips (`…/2026-10-17-refav1-hj-mini/raw/p10_yaw4512/p10_fit.json`, `excluded 540:600, n 60`). Prior and arms see the same data.
  * **(b)** The TRAIN list then equals RA-1 full's. Each run records RA-1's own digests, `train_sha256` and `excluded_sha256` (`ra1_pooled.clips_sha256`), so v8F-mini and RA-1 differ in model, not in data.
  * **(c)** The holdout stays unseen for any later selection.
* Cost: 60 clips (1.3 %).
* Gate: `--exclude` has a G-DVB entry and two read-backs, "60 excluded as declared" and "0 excluded clips in TRAIN". Five source mutations each turn the tests RED (`…/2026-10-17-v8f-mini/raw/full/mutate_exclude.json`).

**3. The residual base AND the reference every H-MINI bar is read against: P10-yaw-4512.**
* Same recipe as Amendment 3 (yaw, κ units, stride 8, seed 0); only the fit clips change, 600 → 4,512.
* MEASURED on eval-141 (2,399 windows, `p10_fit.json`):
  * 6 s ADE **2.4870 m**;
  * FDE 7.667 m;
  * 2 s ADE (first 10 knots) 0.2221 m.
  * Against **2.5817 m** for P10-yaw-600.
* ⇒ **The K-1 bar is 0.0947 m STRICTER than Amendment 3's.** Its interval is produced by this run's `floors` step and reported with the result.

**4. Recipe, frozen now** (the original PREREG is silent on the budget; the handoff's choice is adopted unchanged):
* 4S steps, S = ceil(windows of the 4,512 clips / 64);
* batch 64, lr 3e-4, wd 1e-2, warmup 200, every other trainer default;
* `--kin-mode yaw --tracks cache_full\native_tracks.pt` (4,572 / 4,572 clips);
* grid pool4x10; no `--amp`;
* M2b's IDM at β 0.08, γ 100, as in the arms table.

**5. Arms in this run, in this order:** M1 s0 → M2b s0 → M1 s1 → M2b s1.
* The order is seed-major, so a paired one-seed read exists early. That read decides nothing.
* `M1-navshuf` is evaluated on both M1 seeds (eval-only).
* **NOT in this run:** M2a, M1-navonly, M1-nocons, H-DATA-SCALE-1. Every bar that needs one of them is reported **NOT READ (arm not run)**, never as passed:
  * STRAT-1's "M1 beats M1-navonly";
  * TAC-1's "beats M1-navonly";
  * CONS-1's `M1-nocons − M1`.

**6. How the bars are read here.**
* **H-MINI-K-1** is read for BOTH arms, from `v8f_mini_eval.py compare`.
  * ⛔ No arm is SELECTED on eval-141, the grid that scores K.
  * The original "best(M1, M2a, M2b)" is read as **PASS iff at least one of the two arms passes**, and the 2-test multiplicity is stated with the result.
* **H-MINI-WM-1 (NEW, committed now):** M2b − M1, ADE at 6 s, paired per seed (s0 with s0, s1 with s1).
  * **PASS** iff separated at BOTH seeds, mean < 0, and |Δ| > 2F, where F = max(F_M1, F_M2b).
  * It reads the H-JEPA WM + IDM auxiliary's effect on v8F-mini. It does NOT split WM from IDM; that needs M2a.
  * PASS ⇒ M2a ×2 attributes the gain (PI go).
  * FAIL ⇒ the auxiliary is not a v8F-mini lever at this scale. The WM's remaining route is rung β (the WM as a PLANNER: `HJEPAStack.score_candidates` over the candidate fan).
* **Rule-Zero branches, unchanged from plan §13.6:**
  * K-1 FAIL ⇒ the same arms at `--grid pool8x20` (the queue's `train ARM SEED pool8x20`, in its own run directory). Then more history, then D1, then LoRA.
  * K-1 PASS ⇒ AR-1 (trajectory part) on this rig, and the deferred controls next.
  * Each costs box-2 time and needs the PI's go; the verdict names it.
* **Four families + CONSISTENCY + "acts"**, per run, from each `eval_<tag>.json`, with P10's from `floors_report.json`.
  * STRATEGIC is un-paused for v8F-mini (PI ruling of 2026-10-17), so it is read.
  * Distance-keeping is read through the CONSISTENCY **LEAD** rule (the planned minimum time gap against the `lead_eval139` constraint), with its n.
  * The longitudinal family's own headway / TTC metric is **UNAVAILABLE** here: `v8f_mini_eval.families` does not build its lead-track input. That is a named work item, not a silent drop.
* **Estimator, every interval:** the paired episode-cluster bootstrap over the 141 eval episodes, B = 2,000. It answers **another draw of EPISODES** only. Training-seed variance enters only through F, which with 2 seeds is itself one draw (`H-ESTIM-SEED-1`).
* **Tier:** T1-class open loop.

**7. Operational rules (not scientific bars; they change no reading):**
* box 2 SAFE MODE (`%RUNSAFE%`, below-normal priority, affinity FF, 4 threads), one job at a time;
* the chain starts only after the RA-1 full runs chain prints `ZZRA1FULLRUNS-DONE`, and stops if that chain stopped;
* stop on a reboot (crash), on unreachability, on any nonzero rc;
* stop on the USB-HDD temperature rule of the RA-1 build (≥ 4 of 20 samples ≥ 61 °C, or one ≥ 63 °C);
* stop if a run projects > 10 h after 1,000 steps (the input pipeline would then be the experiment);
* a stopped run resumes from `ckpt_last.pt` on the same argv.
* The verdict is `…/2026-10-17-v8f-mini/code/v8f_mini_full_verdict.py`. It REFUSES mixed provenance: another clip list, a missing holdout, an unfinished run, unequal step counts. Seven source mutations turn its tests RED (`raw/full/mutate_verdict.json`).
* The queue is `code/v8f_mini_box2_queue.cmd`; the chain, guard and job builder are under `code/box2_ops/`.
