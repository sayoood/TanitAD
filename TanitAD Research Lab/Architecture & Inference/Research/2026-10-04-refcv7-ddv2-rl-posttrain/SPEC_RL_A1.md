# SPEC_RL_A1 — amendment A1 to WP-RL (refcv7 DDv2-RL post-training): a stabilised restart

**Written 2026-10-11, BEFORE any A1 training step and before any A1 number.** It amends `SPEC_RL.md`
(sha256 `1ba17160d12e59ad1b7ffaf1de754fa114f405c4db3d8d3e2eea4dc0cb78834f`, unchanged by this file).
Anything this file does not change stays as `SPEC_RL.md` wrote it. The sha256 of this file is
recorded in `raw/SPEC_A1_SHA256.txt` when it is frozen.

The code it binds is the package mirror `code/stack/`:

| file | md5 (A1) | md5 (A0 = Thor live) |
|---|---|---|
| `code/stack/tanitad/rl/ddv2_refcv7.py` | `8532918e85bc0443e69e1665893b6b08` | `701638df39c958d5d1b8580b4c0aef14` |
| `code/stack/scripts/ddv2_rl_refcv7.py` | `de77a163a9178a27bf87c3958a1928af` | `b46d4014128a608b3abaa56c720dc2c1` |
| `code/stack/tests/test_ddv2_refcv7.py` | `e4715bd770da5981842464c496979845` | `b049a992bf5c90aacc3b5e232eb013d7` |

The code cites this file by section: **sec. 2** = warmup / lr, **sec. 3** = KL anchor,
**sec. 5** = kill-switch. The numbering below keeps those references valid.

---

## 1. Why A1 exists — what the A0 run `rl_s0` showed

The full diagnosis is in `RESULT_RL_DIAG.md`. The source for every number is
`raw/thor_rl_s0_metrics_step158.jsonl`: 159 rows, steps 0–158, md5 `0fd63d31cbdf1690e20f83f2a4e6b89a`,
copied from Thor `/home/nvidia/refcv7_post/rl/arms/rl_s0/metrics.jsonl`. All of it is MEASURED.

| 20-step window | step 0–19 | step 140–158 |
|---|---|---|
| `param_delta_norm` median | 0.077 | 5.089 |
| `grad_norm` median | 0.800 | 19.085 |
| `x0_out_of_box_frac_final_a_lon` median | 0.214 | 0.608 |
| `reward_mean` mean | 0.2727 | 0.2126 |

**Verdict: POLICY DRIFT, not a reward or advantage defect.**

* **Grad clip.** The clip of 100 fired exactly once, at step 142 (gn 107.93). It was a spike guard
  (D-5), set about 70× above the healthy maximum. That is a design choice, not a code defect.
* **The run never reached its exit point.** It stopped at step 158 of 2,000, with lr 1.59e-4, still
  inside the 200-step warmup.
* **Why it drifted.** Four contributors:
  * **The lr was not rescaled** when D-11 cut the batch from the paper's 512 to 32 (1/16). The A0 run
    kept 2e-4 (MEASURED, `raw/argv/ARGV_rl_s0.json`).
  * **No anchor held the policy near refcv7:** no KL term, and `il_coef` averaged 0.0262.
  * **`clip_sample` was off** (D-4).
  * **The clip never bound** in the drift regime.

---

## 2. Learning rate and warmup (the code's "SPEC_RL_A1 sec. 2")

| | A0 (`rl_s0`) | **A1** |
|---|---|---|
| peak lr | 2e-4 (the paper's, at total batch 512) | **5e-5** |
| warmup | 10 % of steps = 200 | **`--warmup-steps 100`** (absolute) |
| decay | cosine to 1e-6 at step 2,000 | unchanged |

**Why 5e-5.** The batch is 1/16 of the paper's. Square-root batch scaling gives 2e-4 / √16 = 5e-5;
linear scaling would give 1.25e-5. Square-root is chosen as the less conservative of the two
standard rules, so the run can still move in 2,000 steps. This is a convention (ESTIMATED), not a
measured optimum, and no primary source is banked for it.

**Disclosure — the lr cut alone does not prevent the A0 drift.** The cumulative lr reached by
`rl_s0` at its last step (Σlr to step 158) is **0.01272** (MEASURED from the logged `lr`). The A1
schedule reaches the same Σlr at about **step 305** (ESTIMATED, computed from `lr_at`), and its
total over 2,000 steps is about **0.051**.

A descriptive fit, `param_delta_norm ∝ Σlr^0.736`, has fit window steps 20–158, R² 0.98, n 139.
It is NOT a decision input and must not be extrapolated beyond 2× that range. If drift were a
function of Σlr alone, A1 would reach the A0 drift state around step 305. Sections 3–5 are
therefore the guards; the lr cut is not.

---

## 3. KL anchor to refcv7 (the code's "SPEC_RL_A1 sec. 3")

* **Form.** `--kl-beta 0.04`. At every denoising step i the trainer adds
  `kl_beta * mean_{B,M} KL_i * frac / T`, where `KL_i = lp_ref.detach() − lp_θ`. Both terms are
  evaluated at the reference mean `μ_ref`, under the step's Gaussian with the likelihood std floor
  of 0.1. Analytically this equals `Σ(μ_θ − μ_ref)² / (2·σ²)`, which is ≥ 0 and is 0 at θ = ref.
* **Reference.** The frozen `base` copy, which is refcv7 at 50,400. Its forwards run first, under
  `swapped_params` and `no_grad`, before the θ forwards.
* **Why β = 0.04.** It is INHERITED from GRPO / DPOK practice and was not tuned here. How it scales
  against this loss is UNKNOWN until the KL smoke (sec. 11) logs `kl_m`. **β is fixed for A1; it is
  not tuned on A1 data.**
* **Applies to EVERY arm, RLOFF included.** The KL is added independently of `--rl-weight`, so
  RL − RLOFF still isolates the RL term, and I-2 (`coef_rl_abs_sum == 0` on RLOFF) still holds.
* **New integrity check I-12.** At step 0, θ = ref, so `kl_m < 1e-6` on every arm. A violation makes
  the run VOID: it would mean the reference is not refcv7.

---

## 4. Gradient clip: max-norm 2.0

**Setting:** `--grad-clip 2.0`, down from 100.

**Why this value.** The healthy maximum `grad_norm` over steps 0–39 of `rl_s0` was **1.448**
(MEASURED), and 0 of those 40 steps exceeded 2.0. A clip of 2.0 (1.38× that maximum) therefore
leaves the healthy regime unclipped.

**How it would have bound on `rl_s0`.** First at step 43, and then on 91 of 159 steps (MEASURED).
For comparison, the clip of 100 bound on 1 of 159 steps.

**This is a declared deviation in two senses:**
* from the release, which has no clip;
* from D-5's spike-guard role. D-5 MEASURED that at batch 4 without a clip, 1 of 2 seeds diverged
  (grad norm 15,712).

**How to read it.** The logged `grad_norm` is the PRE-clip norm (MEASURED: step 142 logged 107.93
under a clip of 100). So K1 (sec. 5) still sees drift that the clip is suppressing. The fraction of
clipped steps (`grad_clipped`) is reported as a diagnostic, never as a verdict.

---

## 5. Kill-switch (the code's "SPEC_RL_A1 sec. 5")

The rules are evaluated after every optimiser step on the logged stats (`HALT_KEYS`). Flags:
`--halt-gn-median 5 --halt-oob-median 0.40 --halt-pdn 3.0 --halt-window 20`.

| rule | trips when | rl_s0 replay (MEASURED, `raw/a1_tests/halt_replay_out.txt`) |
|---|---|---|
| K1 | median over the last 20 steps of `grad_norm` > 5 | step 135 |
| K2 | median over the last 20 steps of `x0_out_of_box_frac_final_a_lon` > 0.40 | step 57 (0.4045) |
| K3 | instantaneous `param_delta_norm` > 3.0 | step 104 |
| any | | **first trip at step 57** |

* **What a trip does.** It writes `HALT.json` and `STOP`, checkpoints, and exits. **`HALT.json` is a
  latch:** the trainer refuses to start in that out-dir again.
* **How a trip is reported.** The arm is reported as **FAILED-DIVERGED at step N**, with the rule
  and its value. It is never re-launched under A1.
* **⚠️ The thresholds are POST HOC.** They were chosen after seeing `rl_s0`, against its healthy
  range:
  * `grad_norm` median 0.800;
  * `x0_out_of_box_frac_final_a_lon` median 0.214;
  * `param_delta_norm` 0.077 over steps 0–19.

  They are guards, not endpoints, and no claim is made from them.
* **Pre-registered reading of a K3-only trip.** K3 trips while all of the following hold over the
  last 20 steps:
  * K1 median `grad_norm` ≤ 2;
  * K2 median `x0_out_of_box_frac_final_a_lon` ≤ 0.30;
  * mean `reward_mean` ≥ 0.2727, the `rl_s0` step 0–19 mean.

  This is read as **HALTED-K3-ONLY**: the anchor may be too tight rather than the run diverging. The
  next lever is an A2 amendment that raises K3, written before relaunch. It is not a result.

---

## 6. Exit bars — CARRIED UNCHANGED from `SPEC_RL.md` §6

`SPEC_RL.md` §6, §6.1, §6.2 and §6.3 apply to A1 **verbatim**. That covers:
* the estimator: `paired_episode_cluster_bootstrap`, n_boot 2000, seed 0, FULL-SET means;
* the inference and training floors;
* the variance statement;
* the T1 self-action open-loop PRIMARY on S2 `ade_m`, with all four families. STRATEGIC is NOT
  APPLICABLE, n = 0;
* H-WPRL-1, H-WPRL-2 and H-WPRL-3, the family harm guard, and the S6 non-regression;
* the secondary reads, including the ≥ 60 % fan non-collapse gate;
* both committed outcomes.

**No bar was found ill-posed by the A0 run**, so none is changed. The A1 arms replace the A0 arms
one for one in every contrast:

| A0 arm | A1 arm |
|---|---|
| RL | `rl_a1_s0` |
| RLOFF | `rloff_a1_s0` |
| RL-SHUF | `rlshuf_a1_s0` |
| RL-s1 | `rl_a1_s1` |

BASE is unchanged (refcv7 at 50,400).

**One reading is made explicit: I-4.** It is fixed here, before A1 data, because the A0 log can be
read two ways.

* **The written text** is "human NC = 1 and human DAC = 1, each on ≥ 95 % of training windows". It
  is evaluated **pooled over all training windows seen**.
* On `rl_s0` (MEASURED, 5,088 windows) that pooled reading gives NC 0.9953 and DAC 0.9756, so
  **PASS**.
* The per-step distribution is reported alongside but does not change the verdict. On `rl_s0`,
  human DAC = 1 on ≥ 95 % of a step's 32 windows held on only 82.4 % of steps, minimum 0.875.
  Human NC held on 98.1 % of steps.

**SPEC_RL's exit criterion and the A0 run.** The exit criterion is the T1 panel above, run on the
exports of all four arms. **The A0 run did NOT reach it:**
* it stopped at 159 of 2,000 steps, all inside warmup;
* there was no export and no T1 roll;
* RLOFF, RL-SHUF and RL-s1 never ran.

**No number from the A0 run is a lever effect:** it is one seed, 159 steps, and inside warmup.

---

## 7. Arms and argv

* **Arms.** Four, in the order `rl_a1_s0` → `rloff_a1_s0` → `rlshuf_a1_s0` → `rl_a1_s1`. Each
  differs from `rl_a1_s0` only in its named variable (I-11 unchanged).
* **A0 flags kept.** Every A0 flag of Thor's `wprl_argv.py` (md5 `7f3644414193c23725fd4e4fb238a5a1`)
  is kept, EXCEPT these, which are identical on all four arms:
  ```
  --lr 0.00005 --warmup-steps 100 --grad-clip 2.0 --kl-beta 0.04
  --halt-gn-median 5 --halt-oob-median 0.40 --halt-pdn 3.0 --halt-window 20
  ```
* **Out-dirs.** `/home/nvidia/refcv7_post/rl/arms_a1/<arm>`. These are new directories, so no A1 arm
  can resume an A0 `ckpt_latest.pt`.
* **Trainer.** The A0 trainer rejects the new flags as unknown. A1 therefore needs the A1 trainer
  shipped to Thor's tree first (sec. 11).
* **Unchanged.** B = 32 (4 × 8), K = 2,000, G = 4, `--il-form matched_anchor`, `--dac-rule R2`, the
  windows file, and the segment and checkpoint protocol of SPEC_RL §9.

---

## 8. Resume or restart — RESTART from `ckpt_50400.pt`

**Restart; do NOT resume `rl_s0`.** Three reasons:
1. **Its latest checkpoint is the drifted policy:** `param_delta_norm` about 5, out-of-box about
   0.6, reward down.
2. **Its AdamW moments belong to the 2e-4 regime.**
3. **Resuming under changed flags breaks the A0 run hash, I-11, and the PASS-token argv sha256
   binding.**

**The start point.** `ckpt_50400.pt`, md5 `b418d0fc4a92a6848c246a6a7c50207b`. It is at Thor
`/home/nvidia/refcv7_post/rl/ckpt_50400.pt` (the trainer's `--ckpt` default) and on the dev box at
`D:/refcv7_eval_kit/ckpt/ckpt_50400.pt`.

**⛔ Before ANY restart of the Thor queue, the Master Mind must write two stop files:**
* `/home/nvidia/refcv7_post/rl/arms/rl_s0/STOP`;
* `/home/nvidia/refcv7_post/rl/QUEUE_STOP`.

Why: `rl_s0` has no STOP, HALT or DONE file. A plain queue restart would therefore RESUME the drifted
A0 run under its old argv. The A0 out-dir is kept as evidence and is never deleted.

---

## 9. Cost

| term | value | class |
|---|---|---|
| s / step, A0 config (median over the run) | 31.54 (capture 6.855, rollout 2.15, reward 13.038, grad 9.493) | MEASURED, `raw/thor_rl_s0_metrics_step158.jsonl` |
| KL overhead (one extra no-grad reference forward per micro-batch) | about +10 % (range +10–20 %) | ESTIMATED (grad_s / 3 = 3.16 s) |
| s / step, A1 | about 34.7 | ESTIMATED |
| A-1 condition (s / step ≤ 43.2) | holds, so K = 2,000 stays | ESTIMATED |
| per arm, 2,000 steps | about 19 h of Thor GPU | ESTIMATED |
| 4 arms | about 77 h, plus lock waits | ESTIMATED |

The measured 31.54 s/step is above A-1's ESTIMATED 27.0, but within its 43.2 s condition. A halted
arm costs less.

**The order is the cost gate.** `rl_a1_s0` runs first. If it is halted (sec. 5), the queue stops,
because `wprl_queue.sh` exits when an arm is not DONE. The next lever (sec. 10) is then taken before
any GPU is spent on RLOFF, RL-SHUF or RL-s1.

---

## 10. Rule Zero — the next lever for each A1 failure

`SPEC_RL.md` §10 is carried over. These rows are added for A1's own failure modes:

| observed | next lever | cost |
|---|---|---|
| K1 or K2 trips despite KL + clip 2.0 | **A2: β × 5 (0.2)**, plus an IL floor (`il_coef` ≥ 0.1, or `release_all_modes`) | 1 arm |
| K2 trips with K1 quiet: x̂0 leaves the box | `--clip-sample` (the D-4 x̂0 clamp, currently off) | 1 arm |
| no halt, but the reward does not rise (`frac_positive_after_bar` about 0.0177, as in A0) | rebalance the veto mass: `frac_constraint_fail` was 0.634 on A0, rising 0.62 → 0.70. Then SPEC_RL §10 exploration / larger G | 0 GPU census + 1 arm |
| HALTED-K3-ONLY (sec. 5) | A2: raise K3, written before relaunch | 1 arm |

---

## 11. Launch-gate preconditions — all must hold before step 1

1. **KL wiring smoke on a GPU.** At most 20 steps, under the lock, on `rl_a1_s0`'s argv, in a
   scratch out-dir. It must show:
   * I-12 (`kl_m < 1e-6` at step 0);
   * `kl_m > 0` and finite by step 20;
   * `grad_norm` finite;
   * `HALT.json` absent.

   The end-to-end KL wiring is **UNVERIFIED** until this runs; CPU unit tests are not end-to-end.
2. **Thor tree sync.** The three A1 files are shipped into `/home/nvidia/refcv7_post/rl/tree/stack/`
   and md5-verified against the table at the top of this file. A real `import` of the trainer must
   then show `--kl-beta` in its `train --help`.
3. **The two stop files of sec. 8** are written before any queue restart.
4. **A new launch-gate PASS token from the Master Mind,** binding the sha256 of each A1 argv
   (PI 2026-09-26). The A0 token does not cover A1.
5. **Compute.**
   * Thor is busy with the R2 ladder for about 2–2.5 days (INHERITED from the brief).
   * Off-Thor GPU needs PI spend approval.
   * Neither is decided here.
