# RESULT_V8F_DAG_OACC: OACC's acceleration term inside DAG's fine-tune recipe. Outcome **LONGITUDINAL FIXED**

Claim row `H-V8F-DAG-OACC-1`. Pre-registration `Project Steering/PREREG_V8F_DAG_OACC.md` with **Amendment 1** (`A1-2026-10-27`, landed
`db6a1a43` before any DAG-OACC T1-kin rollout or value; raw sha256 `44d7e879…`). Read 2026-10-27 22:53 by the pre-registered reader
(`raw/2026-10-27-dag-oacc/t1read/dagoacc_read.py`, coding the amendment; enumeration 0 unplaced cells), all **60 reader controls PASS**.
Output `raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json` (sha256 `9e6bd4f6940fb3e868d4d8aede50cf47c082a3833f014c34bb87ff201c99a894`).
**The unamended table gives the same word** (`outcome_unamended`, descriptive), so the amendment moved no outcome here.

* **Tier.** T1-kin: self-action OPEN loop, m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms. Never "closed loop";
  the lead follows its logged future. R-T0 is at T0 (box 2's standard eval).
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired. It
  answers *"would another draw of EPISODES say this?"*. The control is a different training run (DAG), and the training-run question is
  carried by the two seeds and F = 0.1207 m (`H-ESTIM-SEED-1`). The planner is deterministic, so there is no inference-seed term.
* **Arms (box 2, r2 chain at tip `aeb54468`, both gates PASS).** DAG-OACC_s = DAG_s's argv verbatim plus `--w-acc 49.6969`. W was computed
  once by the `waccoacc` stage on DAG_s0, and C-O3 matched the offline recomputation. Swap share reached 0.1013 (s0) and 0.1008 (s1).
  Checkpoints `67265ec1…` / `90b01474…`. Control = DAG_s, banked (one-flag pair, C-O1).

## 1. The outcome, as the amended table decides it

| | s0 | s1 |
|---|---|---|
| PRIMARY-ACC (accel MAE, DAG-OACC − DAG, < 0 and separated) | **PASS: −1.6243 [−1.9766, −1.3236]** (means 1.2727 vs 2.8970 m/s²) | **PASS: −2.0450 [−2.5628, −1.6227]** (1.4885 vs 3.5335) |
| KEEP (ADE6, DAG-OACC − DAG, upper bound ≤ +0.0604 m) | PASS: −0.0356 [−0.1258, +0.0423] | PASS: **−0.1363 [−0.2057, −0.0731]**, separated-better |
| LON-CLEAR (no longitudinal row of DAG-OACC − P10 separated-worse) | clear | clear |
| Read A D1 (ADE6 vs P10 separated-better, ≤ −2F) | yes: −1.2012 | yes: −1.1231 |
| Read A D2 (no row separated-worse vs P10) | **fails**: yaw-rate, curvature, L2 | **fails**: yaw-rate, curvature, L2 |
| Read A D3 (every family has a separated-better row) | yes | yes |
| crawl rows (A1: DOMINANT also needs them to pass) | **fail**: M1 +0.0568 [+0.038, +0.080] | **fail**: M1 +0.0604 [+0.039, +0.086] |
| R-T0 (beside: T0 ADE6, DAG-OACC − DAG) | −0.0100 [−0.0329, +0.0130] | −0.0214 [−0.0536, +0.0077] |

**LONGITUDINAL FIXED** (PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not DOMINANT, because D2 and the crawl rows fail at both seeds).
Committed next, as amended: *the term is adopted and DAG-OACC becomes the base; next is the lever of the row still open*. The open rows are
crawl rows, so the lever is **DAG-ORC re-based on DAG-OACC**. Its Amendment 1 is landed at `6cd741b6`, before any DAG-ORC run. Under Read A's
own table (`PREREG_V8F_DAG_REREAD` §5) the same rows read MARGIN, NOT DOMINANT.

## 2. The four families (T1-kin; point [95 % CI]; B / W = separated-better / -worse)

| family | row | DAG-OACC − P10 s0 | s1 | DAG-OACC − DAG s0 | s1 |
|---|---|---|---|---|---|
| PRIMARY | ADE6 (m) | −1.2012 [−1.395, −1.019] B | −1.1231 [−1.330, −0.916] B | −0.0356 [−0.126, +0.042] – | −0.1363 [−0.206, −0.073] B |
| LONGITUDINAL | speed MAE | −0.7761 [−0.981, −0.600] B | −0.7030 [−0.909, −0.523] B | −0.1996 [−0.281, −0.124] B | −0.4010 [−0.530, −0.295] B |
| | along-track MAE | −0.6809 [−0.844, −0.521] B | −0.5876 [−0.770, −0.412] B | −0.0527 [−0.153, +0.030] – | −0.1304 [−0.254, −0.044] B |
| | accel MAE | −0.8172 [−1.177, −0.500] B | −0.6014 [−0.906, −0.335] B | **−1.6243 [−1.977, −1.324] B** | **−2.0450 [−2.563, −1.623] B** |
| | DK-VIOL | +0.0017 [−0.023, +0.027] – | −0.0034 [−0.024, +0.018] – | −0.0336 [−0.059, −0.012] B | −0.0084 [−0.030, +0.009] – |
| | DK-HW | −0.3318 [−0.694, −0.024] B | −0.4466 [−0.820, −0.169] B | −0.2475 [−0.458, −0.041] B | −0.2080 [−0.439, +0.016] – |
| | DK-TG | −0.0568 [−0.123, +0.016] – | −0.0724 [−0.138, −0.009] B | −0.0725 [−0.145, −0.012] B | +0.0210 [−0.020, +0.086] – |
| LATERAL | heading MAE | −2.5920 [−3.239, −1.975] B | −2.0539 [−2.709, −1.371] B | **+0.7674 [+0.348, +1.265] W** | −0.4548 [−1.093, +0.101] – |
| | yaw-rate MAE | **+3.4726 [+0.660, +6.331] W** | **+8.6844 [+5.169, +12.574] W** | **+3.5126 [+1.429, +5.878] W** | +0.0292 [−1.762, +1.879] – |
| | curvature MAE | **+0.0854 [+0.058, +0.117] W** | **+0.1352 [+0.089, +0.192] W** | **+0.0241 [+0.011, +0.039] W** | +0.0218 [−0.003, +0.053] – |
| | cross-track MAE | −0.9620 [−1.108, −0.832] B | −0.9742 [−1.115, −0.841] B | +0.0173 [−0.015, +0.051] – | −0.0473 [−0.095, +0.020] – |
| | L1 lateral-accel MAE | −2.4618 [−3.810, −1.292] B | −1.8473 [−2.962, −0.850] B | −0.0296 [−0.308, +0.258] – | −0.7120 [−1.204, −0.234] B |
| | L2 curvature MAE (GT ≥ 5.2) | **+0.0443 [+0.023, +0.074] W** | **+0.0529 [+0.028, +0.085] W** | +0.0158 [−0.001, +0.041] – | −0.0075 [−0.030, +0.014] – |
| TACTICAL | TAC-LAT | +0.1905 [+0.151, +0.232] B | +0.1972 [+0.160, +0.236] B | −0.0021 [−0.019, +0.015] – | +0.0121 [−0.010, +0.033] – |
| | TAC-LON | +0.1396 [+0.077, +0.200] B | +0.1371 [+0.075, +0.200] B | +0.0108 [−0.017, +0.039] – | +0.0229 [−0.004, +0.051] – |
| STRATEGIC | S-POS@30 / @50 / @80 vs CTRV (m), per arm | DAG-OACC −1.74 / −4.53 / −9.56 B (DAG −1.72 / −4.48 / −9.69) | DAG-OACC −1.83 / −4.61 / −9.92 B (DAG −1.87 / −4.90 / −10.33) | not computed: no paired S-POS instrument | |
| crawl | M1 crawl share | **+0.0568 [+0.038, +0.080] W** | **+0.0604 [+0.039, +0.086] W** | **+0.0102 [+0.003, +0.017] W** | −0.0011 [−0.007, +0.005] – |

STRATEGIC is reported, not paused, for v8F. The T0 four-family leaves (box 2's standard eval, DAG-OACC against DAG) are in
`DAGOACC_READ.json` (`beside.per_seed.*.T0.four_families_T0`). They are descriptive only: no paired T0 four-family instrument exists. K4 is
NOT RUN, because the PREREG scheduled no logged-bucket rollout or fresh-age T0 eval.

## 3. What the numbers say (beside the outcome; none of this moves it)

1. **The acceleration error is fixed, and position comes along.** Accel MAE falls by 56 % / 58 % (2.90 → 1.27 and 3.53 → 1.49 m/s²).
   Against the kinematic floor P10 it flips from separated-worse for DAG (+0.81 / +1.44, `RESULT_V8F_DAGGER2` §2) to separated-better
   (−0.82 / −0.60). ADE6 is kept at s0 and improves at s1 (−0.136, separated). The effect is 2.6–3.2 times the cross-seed spread of DAG's own
   accel row (0.63 m/s²). That comparison is context, with no bar.
2. **Every longitudinal row is clean against P10 at both seeds.** Speed, along-track and accel MAE and headway are separated-better; the
   time gap is better or not separated; DK-VIOL is not separated. Against DAG, speed MAE is separated-better at both seeds (−0.20 / −0.40),
   so the term fixed target speed as well as smoothness.
3. **What stays open is the crawl set, and at s0 the term costs some lateral.** Against P10, yaw-rate, curvature, L2 and M1 are
   separated-worse at both seeds. Against DAG, s0 reads heading +0.77, yaw-rate +3.51, curvature +0.024 and M1 +0.010, all separated-worse;
   s1 reads none of them worse. The acceleration term did not repair the crawl, so NEXT-CRAWL's COMPOSE condition (DAG-OACC − DAG
   separated-better on the crawl rows) is not met.
4. **The hierarchy's decisions are untouched.** TAC-LAT and TAC-LON against DAG are not separated at either seed. The lever acts on O's
   execution, as designed.

## 4. Next (Rule Zero)

* **Adopted.** DAG-OACC is the base.
* **The crawl rows go to DAG-ORC, re-based** (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`).
  * Form: COMPOSE. A = DAG-OACC's argv plus O-RECOVER's perturbation, behind a new default-off composition flag. CTRL = DAG-OACC, banked.
    REP is new.
  * Box 2: about 2.8 h, inside the PI's go of 2026-10-27.
  * Launch blocker: the composition flag and its reachability proofs, in build.
* **Evidence videos.** The PREREG re-renders them only on DOMINANT, so they are not re-rendered on this read.

## 5. Provenance

* **Box 2.** Chain `chain_v8f_dagoacc_r2.sh` ran 20:21–21:51 (`box2/logs/chain_v8f_dagoacc_r2.out`, marker `ZZV8FMINI-DONE`). Each run
  took 41 and 42 min with its T0 eval. The first attempt (`box2/logs/chain_v8f_dagoacc.out`) stopped at its gate asserter before anything
  trained; the r2 asserter is banked at `raw/2026-10-27-oaccflat-queue/r2/`.
* **Dev-box CPU.** Chain `t1read/dagoacc_t1_chain.sh`: 2 / 2 T1-kin rollouts and the reader. DAG's and P10's runs are the banked ones.
* **Reader checks, before launch.** Self-test 287 / 287. Mutation proof 127 / 127 killable mutants killed; 2 are real-run-only branches,
  UNVERIFIED. Real-config dry run `ZZDAGOACCT1-DRY-OK`. Amendment coding: `t1read/AMEND1_CODING.md`.
* **Not in the repo.** Per-episode dumps, tick arrays and checkpoints stay on E: (`v8f-dagoacc-t1/`, `v8f-dagoacc/`), single copy.
