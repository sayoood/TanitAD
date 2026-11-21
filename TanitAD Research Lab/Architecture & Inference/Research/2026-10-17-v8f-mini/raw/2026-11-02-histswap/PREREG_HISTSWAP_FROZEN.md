# PREREG_HISTSWAP: H-HISTSWAP-1, does O follow its own slowed ego history? (zero training, ORACLE diagnostic)

Tier **T1-kin** (self-action OPEN loop, plain causal bucket, nav dead-reckoned). Evidence class MEASURED (ours) once read. Hypothesis **H** is a HYPOTHESIS.
Estimator of every interval: paired episode-cluster bootstrap, B = 2,000, seed 0 (`taniteval.ci` draw stream via `gapdecomp.decompose`) over the 141 episodes (2,399 windows). It answers *would another draw of EPISODES say this*.
X and Y use the SAME checkpoint, so no training-seed variance sits inside X - Y, and the planner is deterministic (the thread-order floor is read by KA1). The two seeds are the rig's replicates, not a variance estimate.

## 1. Question and arm
Context (MEASURED, `dagland/kappa_t1`, `dagland/feas2_t1`): v8F (DAG-OACC, SOUP; s0, s1) under the SG(7, 2) smoother S passes D1 and D3 against P10; D2 fails on **M1 crawl share** alone (ego speed < GT - 2 m/s, `t1hd_read.py:422`);
every decode lever leaves M1 unmoved or worse, and FEAS2's pre-check finds plan-level crawl BELOW P10's, so the excess sits in the EXECUTED speed.
**H (HYPOTHESIS):** O follows its own slowed ego history: once the executed speed lags, the history input says "slow" and O plans slow.

* **Arm X** = Y's flags + `--histswap logged` (tool `v8f_t1_eval_histswap.py`): at every tick the network's ego-history INPUT (`p10f`, `base`, `kin3`, `pose`, `pose_now/_t/_s`, and the causal speed bucket, i.e. every ego-row-built key) is read from the LOGGED clip arrays at that tick.
  The plan is still committed to the EXECUTED state. Unchanged: nav (dead-reckoned from the executed motion), camera, hold ages, S (`P + SG(plan - P)` with P = the prior the network computed in that call, so P now reads the logged history), the P10 floor (never swapped).
  Tick 0 is identical to Y by construction (the executed history is the logged one at t0); the ticks file's `v0` stays the EXECUTED speed and `v0_in` is the speed the network read.
* **Y** = KAPPA-T1's fresh S runs `SREP_*` (reused, pinned in `FROZEN_SHA.txt`), as FEAS2 reused them. **Bases:** DAG-OACC and SOUP at s0 / s1 (4 rollouts `HSW_*`), plus one bit-identity rerun of a Y (`SREP4_DAGOACC_s0`, `--histswap off`).
* ⛔ **ORACLE, not deployable:** the logged history after t0 is the human's. X's absolute rows are NOT performance and are never quoted as such; only X - Y on M1 and the row structure are read.
  ⚠ Consequence for a positive reading: X also leaks the human's realised motion at ticks 1 and 2, so it cannot tell "the executed lag is the cause" from "any human-motion history fixes it". A negative reading is not subject to that leak.

Pre-check on SEEN Y data (disclosed, `raw/PRECHECK_CEILING.json`, `code/precheck_ceiling.py`): the slow GT-fast pairs of Y minus those of P10, by executed-step block [0-10) / [10-20) / [20-30), are DAG-OACC s0 -25 / +583 / +1,158, s1 +6 / +765 / +1,414,
SOUP s0 +24 / +934 / +1,579, s1 -6 / +716 / +1,497. The excess starts after tick 0, so the swap (which cannot change tick 0) is not ceiling-limited. This pattern is expected under H and under a plan that decays with the horizon; it does not discriminate them.

## 2. Pre-registered reading (committed before any HISTSWAP number; the rule is `decide` in `code/histswap_t1_read.py`, tested in `tests/test_histswap_read_rule.py`)
Per cell (base x seed): `better` = the M1 crawl-share row of X - Y is separated and delta < 0 (a share: lower is better). `share closed` = -(X - Y delta) / (Y - P10 delta) on the M1 row; undefined where Y - P10 is not > 0, and an undefined cell counts as 0 in the mean.

| outcome | condition |
|---|---|
| **H SUPPORTED** | M1 X - Y separated-better at **both seeds on both bases (4 of 4)** AND the mean share closed over the 4 cells >= 0.5 |
| **H NOT SUPPORTED** | M1 X - Y separated-better at **fewer than 3 of the 4 cells** |
| **MIXED** | anything else (3 of 4 cells, or 4 of 4 with mean share < 0.5) |

Reported in full, none of them decisions: all 19 Read-A rows for X (vs P10), for X - Y and for Y - P10 (value, CI, separated); `read_a` / `read_xy`; the per-class (launch / stay / other-stopped / moving) split of the M1 row; the curvature partition; the pooled share closed; `swap_bucket_differs / swap_ticks`.
S-POS rows of X mix the logged poses the S head read with the executed pose the extractor frames them in: **UNVERIFIED**, outside the reading.

## 3. Controls (computed before any row; any failure REFUSES the read)
KY Y pins; KA1 the fresh `--histswap off` rerun equals the reused Y (bit-identical, else max dev <= 1e-4, dump and ticks); KS0 tick 0 of every X equals Y's tick 0 (<= 1e-4); KS1 tick 1's executed `pose` and `v0` equal Y's (same tick-0 plan, so the executor ran from the executed state);
KS2 `v0_in` equals `v0` at tick 0 and differs by > 0.05 m/s on > 5 % of later ticks; KF floor and windows equal Y's; K3 and KB1 as FEAS2. Tool tests (`logs/pytest_histswap.log`, 8 passed): off bit-identical to the FEAS2 tool, the swapped tensors equal the logged arrays by direct indexing,
the plan commits from the executed pose, v0 is executed and `v0_in` logged, the flag refuses `--feasible`; three mutations go RED (item builder on the executed dict; commit from the logged pose; causal bucket from executed speeds). Rule tests: `logs/pytest_read_rule.log` (6 passed, one mutation RED).
A 32-window smoke on the real DAG-OACC s0 checkpoint (<= 2 threads, mechanical comparisons only) is run after this freeze and before the chain.

## 4. ESTIMATED prediction (not a result)
P(SUPPORTED) 0.60, P(MIXED) 0.25, P(NOT SUPPORTED) 0.15. Basis: the oracle history carries the human's realised speeds, so M1 is expected to close largely through the swap whatever the mechanism; the 0.15 is the mass on the slowness living in the scene / perception path that no history swap reaches
(the programme's trigger probes found the frozen encoder caps the go / stay cue). Expected, not committed: X's ADE6 improves vs Y by leak.

## 5. Committed next step
* **SUPPORTED:** the recovery fine-tune of O (`DRAFT_RECOVERY_CARD.md`, not frozen) goes to the PI for a box-2 go, together with the leak caveat of section 1 and the leak-free separating arm named as its unknown; no GPU job is started by this card.
* **NOT SUPPORTED:** the slowness comes from the scene / perception path; the next lever is named from the per-class split of the remaining M1 excess: launch-class share >= 0.5 of it means the trigger / encoder path (PI decision, the frozen-encoder probes are closed);
  moving-class share >= 0.5 means a longitudinal mode-commit head (box-2 candidate, PI go); otherwise both.
* **MIXED:** the 3-of-4 or sub-bar cells are named per cell and the leak-free arm is run before any training go.

## 6. Disclosed scope
Same windows, ckpts and flags as Y except `--histswap`. The swap includes the causal speed bucket (a function of the ego speed history); its separate contribution is not attributed here (`swap_bucket_differs` is counted). Nothing here trains, touches box 2, or lands in git (the Master Mind lands).
