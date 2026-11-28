# RESULT_RL_DIAG — why WP-RL `rl_s0` drifted, and the A1 fix

**Date:** 2026-10-11. **Thor GPU used:** zero. **Author:** Arch & Inference agent for the Master Mind.

**Status:** **FAILED-INCOMPLETE.** The A0 RL arm yielded at step 159 of 2,000 while still in
warmup, and its metrics show policy drift.

**Fix:** pre-registered as `SPEC_RL_A1.md` (sha256 `e08d6f28…b50f18`, frozen 2026-10-11T15:12:44Z,
before any A1 number). It is implemented and CPU-tested in the package mirror.

**Not yet:** not run, not landed.

**Source for every MEASURED number:** `raw/thor_rl_s0_metrics_step158.jsonl` (159 rows, steps
0–158, md5 `0fd63d31cbdf1690e20f83f2a4e6b89a`). It is copied from Thor
`/home/nvidia/refcv7_post/rl/arms/rl_s0/metrics.jsonl`.

**Eval tier:** none. These are **training-time diagnostics, not an eval**. No export or roll exists.

---

## 1. SPEC_RL's exit criterion — NOT reached

`SPEC_RL.md` §6 / §6.1 makes the T1 self-action open-loop panel the exit criterion: S2 `ade_m` plus
four families, at inference seeds 0 and 1, over the exports of RL, RLOFF, RL-SHUF and RL-s1, with
H-WPRL-1/2/3.

**159 steps did not reach it:**
* the run stopped at 159 / 2,000 steps;
* lr was 1.59e-4 at step 158, still inside the 200-step warmup;
* there was no export and no T1 roll;
* RLOFF, RL-SHUF and RL-s1 never started.

**H-WPRL-1 / 2 / 3:** NOT TESTED.

**Four families, all n/a** for the same reason: no eval ran, n = 0.

| family | status |
|---|---|
| LONGITUDINAL | not computed — no export |
| LATERAL | not computed — no export |
| TACTICAL | not computed — no export (the heads are frozen identity controls anyway) |
| STRATEGIC | not computed — NOT APPLICABLE per SPEC_RL, n = 0 |

**⛔ No number below is a lever effect.** The run is one seed, 159 steps, all inside warmup, and
has no control arm (H-ESTIM-SEED-1).

---

## 2. What happened (MEASURED)

| 20-step window | step 0–19 | step 140–158 |
|---|---|---|
| `param_delta_norm` median | 0.0767 | 5.0892 |
| `grad_norm` median (mean) | 0.8002 (0.8255) | 19.085 (29.895) |
| `x0_out_of_box_frac_final_a_lon` median | 0.2143 | 0.6077 |
| `reward_mean` mean | 0.2727 | 0.2126 |
| `human_pdms_mean` mean (policy-invariant control) | 0.9569 | 0.9623 |
| `chain_endpoint_spread_m` median | 34.38 | 30.23 |
| `cand_dac_mean` median | 0.450 | 0.363 |

**Grad-norm growth.** Mean ×36.2 (0.8255 → 29.895); median ×23.9 (0.8002 → 19.085).

**First crossings:**
* `param_delta_norm` > 1 at step 46, > 2 at step 69, > 3 at step 104;
* oob median-20 > 0.40 at step 57;
* gn median-20 > 5 at step 135.

**Correlations** (per-step Pearson, n 159):
* reward ~ `param_delta_norm`: −0.615;
* reward ~ oob: −0.597;
* `human_pdms_mean` ~ step: +0.035.

---

## 3. Diagnosis

### (a) "The clip never fired" — it fired ONCE, and it was never meant to bind

* The clip is max-norm 100. The logged `grad_norm` is PRE-clip: step 142 logged 107.93.
* Over the whole run it bound on **1 / 159** steps, at step 142.
* The healthy maximum over steps 0–39 was **1.448**, so the clip sat about 70× above it. That is
  D-5's stated spike-guard role: the release has no clip, and D-5 added 100 against a 15,712
  divergence at batch 4. The clip did what it was designed to do.
* ⇒ **No code defect.** The clip was **structurally non-binding** in the regime that mattered.
  * A clip of 2.0 would have bound from step 43 (91 / 159 steps had gn > 2).
  * No pre-fix / post-fix unit test applies, because no defect exists.
  * The optional GPU smoke (brief item 3) is SKIPPED, as the brief specifies when (1) finds no code
    defect.

### (b) The reward fall is POLICY DRIFT, not a reward or advantage defect

| check | result | reading |
|---|---|---|
| Policy-invariant control `human_pdms_mean` | flat: 0.957 → 0.962, r with step +0.035 | The reward function scores a fixed trajectory the same way throughout. A scorer defect would move this. |
| I-3: `frac_positive_after_bar` > 0 on ≥ 90 % of steps | 159 / 159 | PASS — the advantage is not degenerate |
| I-5: `cand_dac_mean` < 1 on ≥ 50 % of steps | 159 / 159 | PASS |
| I-1: finite, `param_delta_norm` > 0 | 159 / 159 | PASS |
| Reward vs policy output | reward ~ `param_delta_norm` −0.615; ~ oob −0.597 | The reward falls with how far the policy has moved and how often x̂0 leaves the box. |
| Fan | spread 34.38 → 30.23 m (88 %) | not collapsed (≥ 60 % bar) |

**I-4, human NC / DAC = 1 on ≥ 95 % of training windows:**
* Pooled over 5,088 windows: NC 0.9953, DAC 0.9756 ⇒ **PASS** as written.
* Per step, DAC holds on only 82.4 % of steps (minimum 0.875); NC on 98.1 %.
* SPEC_RL_A1 §6 fixes the pooled reading **before** A1 data.

### (c) Why it drifted — the contributors

| contributor | evidence | class |
|---|---|---|
| lr not rescaled for the batch cut | D-11 cut the batch to 32 (1/16 of the paper's 512) and kept the paper's 2e-4 (`raw/argv/ARGV_rl_s0.json`). | MEASURED |
| no anchor to refcv7 | no KL term in the A0 code; `il_coef` mean 0.0262 | MEASURED |
| x̂0 unclamped | `clip_sample` off (D-4); oob rose 0.21 → 0.61 | MEASURED |
| clip non-binding | 1 / 159 | MEASURED |
| reward signal thin | `frac_positive_after_bar` mean 0.0177; `frac_constraint_fail` mean 0.634 | MEASURED; its causal role is UNVERIFIED |

**Drift is monotone in cumulative lr.** Σlr to step 158 = 0.01272. A descriptive fit gives
`param_delta_norm ∝ Σlr^0.736` (fit window steps 20–158, R² 0.98, n 139). It is not a decision
input and must not be extrapolated beyond 2×.

**The mechanism of the grad-norm growth is UNVERIFIED.** HYPOTHESIS: with the likelihood std floor
at 0.1, the log-prob gradient scales with |μθ − sample|, so it grows as the policy drifts. This is
not tested here.

---

## 4. The fix — `SPEC_RL_A1.md` (pre-registered; RESTART, not resume)

| lever | A0 | A1 | why |
|---|---|---|---|
| peak lr | 2e-4 | **5e-5** | √-batch rescale for the 1/16 batch. A convention, ESTIMATED. |
| warmup | 200 (10 %) | **100 steps** | absolute steps, `--warmup-steps` |
| grad clip | 100 | **2.0** | 1.38× the healthy maximum of 1.448. A declared deviation from the release and D-5. |
| KL to refcv7 | none | **β 0.04** on ALL arms | β INHERITED from GRPO / DPOK; new integrity check I-12: `kl_m` < 1e-6 at step 0 |
| kill-switch | none | **K1** gn med20 > 5 · **K2** oob med20 > 0.40 · **K3** `param_delta_norm` > 3.0 | POST HOC thresholds; a trip = FAILED-DIVERGED, latched by `HALT.json` |
| exit bars | SPEC_RL §6 | **unchanged** | none was found ill-posed |

**What the lr cut alone does — disclosed.** It does not prevent the A0 drift: A1 reaches A0's
Σlr = 0.01272 at about step 305 (ESTIMATED). The KL, the clip and the kill-switch are the guards.

**Replay on `rl_s0`** (MEASURED, `raw/a1_tests/halt_replay_out.txt`): K2 would have tripped at
**step 57** (0.4045), K3 at 104, K1 at 135.

**Restart from `ckpt_50400.pt`** (md5 `b418d0fc…`) into new out-dirs `arms_a1/<arm>`. Do not resume
`rl_s0`, for three reasons:
* its policy has drifted;
* its AdamW moments belong to the 2e-4 regime;
* resuming would break the PASS-token / I-11 binding.

---

## 5. Tests (MEASURED, dev-box CPU, against Thor's exact tree)

| suite | result |
|---|---|
| A0 suite on the A0 code | **36 passed** (`raw/a1_tests/pytest_base.txt`) |
| A1 suite on the A1 code | **55 passed** (`raw/a1_tests/pytest_a1.txt`) — 15 new test functions (35 → 50; 55 collected after parametrisation) for the KL, the halt rules, warmup and the parser flags |

**Deliberate-regression arms** (`raw/a1_tests/mutate.py`, output in `mutate_out.txt`): **ALL ARMS
AS EXPECTED**.

| arm | mutation | result |
|---|---|---|
| M1 | KL at t, not t−1 | RED |
| M2 | KL sign flipped | RED |
| M3 | reference params not restored | RED |
| M4 | median → mean in K1 / K2 | RED |
| M5 | K3 waits for the window | RED |
| M6 | `--warmup-steps` ignored | RED |
| M7 | KL gradient reaches the reference | RED |
| M7e | equivalent mutant (only `x0_ref.detach()` removed) | GREEN, documented |

**Expectations are literals,** e.g. analytic KL `Σ(Δμ)²/(2σ²)` and literal halt reasons.

⚠️ **The CPU tests are unit-level only.** The end-to-end KL wiring on the GPU is **UNVERIFIED**
until the ≤ 20-step smoke in SPEC_RL_A1 §11.1.

---

## 6. Rule Zero — the next arm, and what blocks it

**The next arm is `rl_a1_s0` under SPEC_RL_A1. It cannot run in this turn.** Each blocker, and what
unblocks it:

1. **Thor GPU:** busy with the R2 ladder for about 2–2.5 days (INHERITED from the brief).
   *Unblock:* the ladder finishes, or the PI approves spend for an off-Thor GPU.
2. **KL GPU smoke** (≤ 20 steps; I-12): needs a GPU. On the dev-box RTX 4060 it would go through
   `with_gpu_lock.py`, but it was not attempted, because the brief limits item 3 to a code-defect
   finding. *Unblock:* Master Mind authorises a dev-box smoke, or folds it into the first Thor
   segment.
3. **A1 not landed:** A0 is landed (§7); `LANDING_READY_WPRL_A1.txt` is not, and the A1 edits exist
   only in this working tree. *Unblock:* the Master Mind's lander.
4. **Thor queue would RESUME the drifted `rl_s0`:** it has no STOP, HALT or DONE file. *Unblock:*
   the Master Mind writes `/home/nvidia/refcv7_post/rl/arms/rl_s0/STOP` and
   `/home/nvidia/refcv7_post/rl/QUEUE_STOP` before any restart.
5. **Thor tree is stale:** `/home/nvidia/refcv7_post/rl/tree/stack/` holds the A0 code, which
   rejects the A1 flags. *Unblock:* ship the 3 files and md5-verify them.
6. **A new launch-gate PASS token** binding the A1 argv sha256s is needed (PI 2026-09-26).
7. **The A1 argv helper (`wprl_argv_a1.py`) was not written.** The file write was refused by the
   tool's auto-mode classifier. The full argv delta is in SPEC_RL_A1 §7.

**If A1 fails its bar,** SPEC_RL_A1 §10 names the next lever per failure mode:
* β × 5;
* an IL floor;
* `--clip-sample`;
* a veto-mass rebalance;
* raising K3.

---

## 7. Provenance — A0 is landed; A1 edits it

A0 is already on `origin/agent/arch-inf-20260803`, tip `4d937212`. The package landed in
`0c618136` (2026-10-04); SPEC_RL.md matches sha256 `1ba17160…`. Checked by blob comparison
against Thor's tree:
* `stack/tanitad/rl/ddv2_refcv7.py` = `fefb9b99…` — identical to Thor's;
* `ddv2_rl.py` `319ad19b…`, `pdm_proxy.py` `9ba617da…` and `models/kinematic_prior.py` `3438b8f7…`
  are also identical to Thor's;
* the package mirror `code/stack/*` holds the exact A0 bytes, `701638df…` / `b46d4014…` /
  `b049a992…`.

⚠️ The local checkout's HEAD (`37645fcc`) is **behind** origin. That is why `kinematic_prior.py`
looked missing in "the main repo": it is present at the origin tip.

The A1 files are **edits over landed A0 files**:
* mirror md5 `8532918e…` / `de77a163…` / `e4715bd7…` (CRLF);
* the lander applies the same content to `stack/…`.

The new flags all default to A0 behaviour (`--warmup-steps 0`, `--kl-beta 0`, `--halt-* 0`), so
existing A0 argv is unaffected.
