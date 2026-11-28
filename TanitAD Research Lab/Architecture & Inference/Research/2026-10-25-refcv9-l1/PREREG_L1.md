# PREREG_L1 — refcv9 L1: does a progress head on refcv8's `z_tac` beat the ego-scalar floor along refcv8's own path?

*Architecture & Inference agent, 2026-10-25 (Berlin). Sealed BEFORE any 48,852 `z_tac` exists: the sha256 of this file is written to
`PREREG_L1.sha256` and checked by `code/l1_ztac_roll.py --require-seal` before every scored roll. Source of the experiment:
`Project Steering/refcv9/L1_DESIGN_ANNEX.md` §2.4 (mechanism c4) and §4 (E1 ranking run); `Project Steering/PLAN_REFCV9.md` §0, §2 L1.*

## 0. Identity

| item | value |
|---|---|
| model | refcv8 EMA-48,852, `E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt`, md5 **b4364029b68f29a33eeb4aebbbbd002b**; config `E:/refcv8_eval_kit/ckpt/config.json` (`--no-strategic`, `--residual-prior ha0_ext_pose`) |
| eval windows | the SHIP-R8 battery's eval139 grid, `C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/windows_s{0,1}.json`, md5 **a3a78dba889e70aa69a05271559794ad** (both), 4,754 windows / 139 episodes, per-frame nav feed (`perframe_full`) |
| banked plans scored | that tree's `dump_s0` / `dump_s1` (inference seeds 0 and 1) = refcv8's own selected, post-cap emitted plans |
| train windows | A6 train139 (139 clips, fit guard: 0 overlap with eval139), registered window rule `t % 5 == 0` (4,855 windows, the SAME fit set the shipped RETIME table was fit on); target + scalar features read from the model-free part of `E:/refcv8_eval_kit/trainfit/step30000ema_train139` (prior, GT, pose window, fed ceiling: identical for every checkpoint) |
| shipped stage (H0 table) | `…/2026-10-20-refcv8-perframe-final/raw/retime_trainfit_pf/FROZEN_GAINS_RETIME_train_pf.json` (the per-frame refit, SHIP-R8 Amendment 3 (ii)); its sha256 is recorded in the result |
| launch tree | rebuilt `C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58` (fb47a583763f); the original `C:/lgs/Lfb47a58376` is gone |
| code at seal | `code/l1_ztac_roll.py`, `code/l1_fit_score.py`, `code/l1_verdict.py` (the mechanical §7-8 reading), `code/chain_l1.sh`, `code/train_shim/*`, the probe copies in `code/probe_repro/`; sha256 list in `CODE_SHA256_AT_SEAL.txt` |

## 1. Question

refcv8's emitted plan has the right path shape but a noisy along-track speed choice (annex §1: residual over the prior ρ 0.84, alignment r 0.38).
The shipped re-timing stage shrinks that noise (E1 −0.034 at 48,852 per-frame, MEASURED, `FORWARD_DEPLOY8_step48852ema_pf.json`), and an
ego-scalar ridge ties it (−0.0328 vs −0.0318 at 30k, MEASURED, `l1_probe/PROGRESS_SCALAR.json`). **L1 asks whether refcv8's own tactical
latent `z_tac` carries progress information the scalars do not** — i.e. whether a deterministic progress head reading `z_tac` beats the
scalar floor, separated, along refcv8's own selected path. Lateral is untouched by construction (every arm keeps the path; only arc length
changes).

## 2. The bank (GPU, dev-box RTX 4060 only; Thor untouched)

`code/l1_ztac_roll.py` wraps the battery children unchanged (r8oracle / P4 pattern) and copies the input of the model's single
`lat_head_tac` module (= `z_tac`, `refs/refc_v3.py:1737-1743`; under `--no-strategic` z_tac = φ_tac(pooled image sequence) + nav-token
term + max-speed-ceiling term; `ego_state_inject` is false in this build, so no curvature enters) for the nav_true row of every window.

| roll | windows | seed |
|---|---|---|
| eval | 4,754 (eval139) | inference seed 0 |
| eval | 4,754 (eval139) | inference seed 1 |
| train | 4,855 (A6 train139, `t % 5 == 0`) | inference seed 0 |

**Price (ESTIMATED before data, d_tac = 512 per `refs/refc_v3.py:365`, UNVERIFIED for this build until the bank's meta reads it):**
14,363 windows × 512 × 2 B (fp16) = **14.7 MB** (stored fp32: 29.4 MB) + three os-only roll dumps (~45 MB each, MEASURED size of the
banked tree's dumps). Total new disk on C: **< 0.5 GB**, far under the 8 GB cap. Storage does not bind; the host commit limit does
(each roll needs a free-commit gate, §9).

## 3. Splits — nothing is tuned on eval

* **Fit split = TRAIN only**: the ceiling-compliant S2 rows of A6 train139 (4,211 windows / 137 episodes at 30k; the compliance mask is
  model-free so it is identical here). Every hyper-parameter (λ, PCA k, gain-model λ, MLP weight decay / epochs) is chosen by **5-fold
  EPISODE cross-validation on TRAIN rows** (fold assignment numpy seed 0; replicate seed 1). Standardisation and the PCA basis are fit
  inside each CV fold on the fold's train part.
* **Scored split = eval139 ceiling-compliant S2 rows** (4,159 windows / 138 episodes, MEASURED in `FORWARD_DEPLOY8_step48852ema_pf.json`),
  scored at both inference seeds on that seed's banked os plan. Never used to choose anything.
* Disclosure: A6 train139 is IN-SAMPLE for refcv8 (it trained on these clips). The head's target (human progress) is model-free, but
  `z_tac` on train clips is an in-sample latent; a head fit on it can read optimistic on train. The eval score is the only quotable one.

## 4. Arms (all on S2 instants 0.5 / 1.0 / 1.5 / 2.0 s = model slots 0–3; the path = polyline (0,0) → refcv8's emitted S2 points, end ray)

Notation: s_P = arc length of the causal prior `ha0_ext_pose`; s_H = the human's; h = s_H − s_P (the target); ĥ = a head's prediction;
re-timing = `progress_headroom.at_progress(O, s')` (the shipped stage's geometry, `retime_stage.polyline_geometry`).

| arm | definition | role |
|---|---|---|
| **OS** | refcv8 as emitted | reference |
| **H0** | s' = s_P + g_k (s_os − s_P), g = the per-frame frozen S2 RETIME table | **refcv8 + shipped stage** (= mechanism (a)) |
| **H1** | g_k(x) = clip(a_k + b_k·x̃, 0, 1.5), x̃ = (v0, a0, ω0) standardised on train; fit by ridge LS of h_k on [d_k, d_k·x̃] with d_k = s_os − s_P on refcv8-48,852's TRAIN os plans (bank, seed 0); λ by CV | mechanism (b) (secondary) |
| **H2s** | ĥ = ridge on the 10 scalar features of `progress_scalar.py` (1, v0, a0, ω0, v0·ω0, v0², a0·v0, min(v_lim,40), min(v_lim,40)−v0, v_lim valid), λ by CV — the probe's procedure unchanged | **mandatory scalar floor** |
| **H2** | ĥ = ridge on [the same 10 scalars ‖ PCA_k(standardised z_tac)], (k ∈ {4, 8, 16, 32, 64, 128, 256}, λ ∈ {1e-2 … 1e4}) by CV | **the lever (primary)** — a linear progress head; same class and loss as H2s, so H2 − H2s isolates the input |
| **H2n** | the annex's head: [scalars ‖ z_tac] → Linear(·,128) → GELU → Linear(128,4), **L1 loss**, AdamW lr 1e-3, batch 256, weight decay ∈ {1e-4, 1e-2, 1e-1} and epochs ∈ [1, 200] by CV (summed held-out L1), refit on all train, torch seed 0 | the annex's MLP form (secondary, pre-registered) |
| **H2m** | H2's fitted head applied with eval z_tac **deranged across episodes** (episode e receives the next episode's z_tac rows, sorted ids, cyclic, position-matched) | **deliberate regression** |
| **H2c** | H2 with z_tac := its train mean | collapse check (beside) |
| **H2r** | H2's whole recipe refit on a random 80 % of train EPISODES (numpy seed 1), CV folds seed 1 | **head-level replicate** |
| **H2g** | ĥ_g = c_k·ĥ (H2), c_k = ⟨ĥ_oof, h⟩/⟨ĥ_oof, ĥ_oof⟩ from H2's out-of-fold train predictions | O3 composition (secondary) |
| **CONST** | ĥ = train mean of h | **constant-only control** |
| H2_PRIOR, H2s_PRIOR | the same heads on the PRIOR's own path | beside (model-free path + vision progress) |

## 5. Metrics, estimator, tier, inference-seed floor

* **E1 cell** = S2 ADE(arm) − ADE(`ha0_ext_pose`) (D-R8-E1-REF), per window, ceiling-compliant rows.
* **Four families** (every arm pair H2−H0, H2−H2s, H2−OS, H0−OS), the battery's `pair_read_r8.METRICS` rows via `pair_read_r8.components`:
  ADE (ade, fde) · LONGITUDINAL (speed MAE, along-track MAE, |along| at 2 s, target-speed within ±0.5 m/s, accel MAE) · LATERAL (heading
  MAE, curvature MAE masked, yaw-rate MAE masked, cross-track MAE, |cross| at 2 s) · TACTICAL (trajectory-derived lateral / longitudinal
  decision correct). **Not computed here, stated per family:** LONGITUDINAL distance-keeping (headway / TTC to the lead agent): this
  CPU re-timing harness reads no lead-agent track, n = 0 — a named work item; STRATEGIC: refcv8 runs `--no-strategic`, n = 0.
* **Estimator**: `pair_read_r8.paired_cell` → `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, seed 0, over EPISODES; it answers
  *another draw of episodes* only — not another training run (H-ESTIM-SEED-1), not another inference run. *Separated* = the 95 % interval
  excludes 0.
* **Inference-seed floor** F_os = |paired ADE(OS, s0) − ADE(OS, s1)| on the same rows (point and CI); every effect also reports
  |Δ_s0 − Δ_s1|.
* **Tier**: T1 self-action open loop on logged eval139 frames. Every re-timed arm is a **post-hoc emulation** of the in-forward layer,
  applied to the POST-cap emitted path; on windows where the `plan_vmax_clamp` cap binds, the in-forward layer (which sits before the cap)
  can differ — UNMEASURED here, reported as a caveat, never folded in.

## 6. Controls (known values; any failure ⇒ INVALID, §8 O5)

| id | control | must read |
|---|---|---|
| K1 | H0 reproduces the banked per-surface `RETIME_TRAINFIT` E1 cell of this tree (`pfpkg/raw/retime_trainfit/FORWARD_TRAINFIT_step48852ema_pf.json`) | equal to ≤ 1e-4 m at both seeds |
| K2 | H0 with g ≡ 1 = OS bit-for-bit; zero correction on the prior's path = the prior | max abs 0 / ≤ 1e-5 m |
| K3 | CONST's held-out eval R² per instant (analytic: a constant cannot have positive R² about the eval mean) | ≤ 0 at every instant |
| K4 | the bank's OWN plan = the banked battery dump's `nav_true.traj`, every eval window, each seed; and its slots 0–3 = the panel's os rows | max abs ≤ 1e-5 m |
| K5 | bank alignment: (e_i, t + W − 1) of the bank = the dump extras' (clip_index, ws), same set; train rows all join; v0 fed = panel v0 | equal; ≤ 1e-3 m/s |
| K7 | H2s's λ and train R² on S2 equal the 30k probe's (`PROGRESS_SCALAR.json`: λ 10, train R² 0.51 / 0.10 / 0.07 / 0.10) — the target, features and fit rows are model-free and identical | λ equal, R² equal to 1e-6 |
| V-m | the deliberate regression H2m must NOT beat H2s separated at either seed | not separated below |
| K6 | z_tac inference-seed dependence max \|z_s0 − z_s1\| / rms(z) (z_tac is upstream of the sampler) | reported (expected ≈ 0); not a bar |

## 7. Bars (both inference seeds, S2 compliant, mechanical)

* **P1** H2 − ha0_ext_pose (E1) separated < 0 at both seeds.
* **P2** H2 − H0 separated < 0 at both seeds (beats refcv8 + stage).
* **P3** H2 − H2s separated < 0 at both seeds AND |Δ| > F_os at each seed (vision adds over the scalar floor).
* **P4** |E1(H2) − E1(H2r)| < |E1(H2) − E1(H0)| at both seeds (the head-level replicate is closer to H2 than H2 is to the floor).
* **G (B9-0 guard)** for H2 − H0: no METRICS row (ADE / LONGITUDINAL / LATERAL / TACTICAL) separated WORSE at both seeds (worse = CI above 0 for error rows; CI below 0 for the higher-is-better rows `TAC_traj_*_correct`, `LON_speed_within_0p5`).

## 8. Committed outcomes (written before data)

* **O1** (K-all ∧ V-m ∧ P1 ∧ P2 ∧ P3 ∧ P4 ∧ G) → **L1 = (c4) with `z_tac`.** refcv9 trains the head jointly, stop-grad on z_tac first
  (every other head bit-identical); full-grad only with a second training seed (B9-2). Next: the in-forward `ProgressRetime(mode="head")`
  port + its pre-cap equivalence check.
* **O2** (valid, P3 fails) → **the frozen latent adds nothing over the scalars at this head class and fit size.** L1 = (a)/(b) in-forward;
  refcv9 cannot beat refcv8 + stage on E1 through a frozen-`z_tac` head, reported as such. Next levers, in this order: (i) H2n if it
  clears P3 at both seeds (a pre-registered secondary, so it needs its own replicate before it is quoted); (ii) a TRAINING-side progress
  head (full-grad, the trunk learns speed cues) as a refcv9 ladder arm with a replicate; (iii) (c3) selection-side on the ladder / L3
  K-draw distillation (PLAN §2).
* **O3** (valid, P3 passes, P2 fails) → compose: read H2g (computed in this run); if H2g passes P2 at both seeds the composition is the
  L1 form, else L1 = (a)/(b) and the head is a diagnostic only.
* **O4** (valid, P1–P4 pass, G fails) → the named family is the next diagnosis; nothing ships.
* **O5** (any K fails or V-m fails) → **INVALID**: no outcome is read; the failing control is the next work item.

## 9. Execution, budget, storage

* GPU chain `code/chain_l1.sh`: identity checks (this seal, ckpt md5, windows md5, shim byte-identity) → GPU quiet (no python compute app)
  → free commit ≥ 6 GB (wait ≤ 3 h, else STOP and report the blocker) → train smoke (1 clip, 3 windows) → eval s0 → eval s1 → train s0,
  each asserted on its ARTIFACT (final bank npz with the expected window count), opaque `ZZL1…ZZ` markers in `chain.log`.
* **Budget**: ≤ 6 GPU-h ESTIMATED (rolls MEASURED at ~0.57–0.60 s/window os-only: ~2.4 h for the three). **Registered subset** if the
  first roll projects > 6 h in total: drop the eval seed-1 roll; seed 1 is then scored with seed 0's z_tac against seed 1's banked plan
  (valid only if K6 on a 100-window seed-1 roll reads ≤ 1e-3 relative).
* Storage: everything under `C:/Users/Admin/refcv8_pf/refcv9_l1/`; nothing written to E:, F: or G:. No git.
* **Secondary, beside, never binding:** H1, H2n, H2c, H2g, the PRIOR-path variants, and **S6 (E3, instants 1–6 s)** with S6-fit heads by the
  same recipes on the train S6 panel rows (K1 on S6 as on S2).

## 10. Deviations

Code fixes that change no definition above (crash, path, dtype) are allowed after the seal and are logged in `DEVIATIONS.md` with the
reason and the diff summary. Any change to an arm, a bar, a split or a grid is a deviation that makes the affected reading exploratory.

## 11. Pipeline validation done BEFORE the seal (no 48,852 data touched)

* `code/probe_repro/progress_scalar.py` (the annex probe, unchanged) re-run on the rebuilt tree reproduces the banked
  `PROGRESS_SCALAR.json` exactly (EMA-30k: PROG_SCALAR_OS −0.0328 / −0.0323, PROG_SCALAR_PRIOR −0.0212, RETIME −0.0318 / −0.0313, λ 10, n 4,211).
* `code/l1_fit_score.py --dryrun-synthetic` on the **EMA-30k** tree with a **Gaussian-noise `z_tac`** (d 512): every control reads its
  value (K2, K3, K5, K7 PASS; V-m holds), and the noise latent FAILS P3 as it must (H2 − H2s +0.0004 / +0.0008 m), outcome O2.
  Output `dryrun_synthetic/` (VERDICT.json, VERDICT_TABLES.md). This tests the fitting/scoring code, not the lever.
