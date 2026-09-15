# PREREG — DiffusionDriveV2's RL stage on refcv5-v2, dev-box validation

**Written:** 2026-09-15, **before any arm ran**. **Owner:** Arch+Inference implementation/validation agent.
**PI request:** *"Prepare and validate the RL approach as described in the paper"* (DiffusionDriveV2) — dev box for preparation, a pod later for heavy work.
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-ddv2-rl-prep/` (SPEC, DIFF, RESULT, raw).
**What ran BEFORE this file, and why it does not contaminate the test:** (a) the zero-training diagnostics `raw/ddv2_diagnose.json` (12 RL-train windows, no update applied); (b) a 4-step smoke per arm on RL-train windows (machinery check; those checkpoints are never scored); (c) a 2-episode T1 harness load test of the smoke checkpoint on held-out clips (loader check; its numbers are not read). No held-out metric of any trained arm has been looked at.

## 1. Hypotheses

* **H-DDV2RL-1 (primary, T0).** On refcv5-v2 (cold start `C:/Users/Admin/refcv5v2_final/ckpt.pt`, md5 `9405ec73b2d797c4cebd44f82dbce54b`, step 40,284), adding DiffusionDriveV2's RL term — released arithmetic, `SPEC_DDV2_RL_PAPER.md` — raises the PDMS-shaped proxy reward of the **deployed** sampler's fan on held-out clips, relative to a control that differs **only** by removing the policy-gradient term.
* **H-DDV2RL-2 (harm guard, T1 = self-action OPEN loop).** The RL arms do not make the deployed selected plan worse than the cold start on the T1 panel.

Nothing here claims PDMS (the reward is a proxy), closed-loop driving (T1 is open loop by the PI's 2026-09-02 ruling), or drivable-area compliance (no map on the RL-train split).

## 2. The machinery, frozen

| piece | file | pinned by |
|---|---|---|
| step, chain, advantage, loss (released arithmetic) | `stack/tanitad/rl/ddv2_rl.py` | `stack/tests/test_ddv2_rl.py` — bitwise vs the vendored release scheduler and rollout loop; per-step backward == one-graph loss |
| refcv5-v2 binding | `stack/tanitad/rl/ddv2_refc_chain.py` | `stack/tests/test_ddv2_refc_chain.py` — bitwise parity with `_sample`; **D1 on the real checkpoint: 12/12 windows bitwise** |
| reward proxy | `stack/tanitad/rl/pdm_proxy.py` | `stack/tests/test_pdm_proxy.py` — analytic scenes + regressions |
| driver | `stack/scripts/ddv2_rl_refcv5.py` | the harness's own `load_model` / `build_corpus` / conditioning |

## 3. Arms

| arm | command (all: `--steps 600 --batch 4`, defaults otherwise) | differs from RL-s0 by |
|---|---|---|
| **BASE** | the cold start, untrained | no training |
| **RL-s0** | `train --arm rl --seed 0` | — |
| **NORL-s0** (length-matched control) | `train --arm norl --seed 0` | the policy-gradient coefficient is **0** on every step; the IL weights stay the release's advantage-derived 0.1 / 1.0 (see below). Same window order and the same chain noise at step 0 |
| **RL-s1** (replicate) | `train --arm rl --seed 1` | window order and chain noise |

⭐ **Why NORL keeps the advantage-derived IL weight.** The release sets λ = 0.1 on rows with a positive advantage and 1.0 elsewhere (`rl.py:1113-1117`). Zeroing A would move λ to 1.0 on every row — MEASURED D6/smoke: every cold-start row carries a positive, so the RL arm runs at λ ≈ 0.1 — and would confound "REINFORCE on/off" with a 10× IL-weight change.

**Common settings (release constants unless stated):** G = 4 × 117 anchors = 468 chains; labels [18 … 0] with one-unit transitions; η = 1; exploration floor 0.04; likelihood floor 0.10; **both release clamps ON** (decoder-input clamp and `clip_sample`); γ = 0.8; ≥GT mask; −1 veto on NC ≠ 1 (DAC ≡ 1, no map); AdamW lr 2e-4, wd 1e-4; 10 % linear warmup then cosine to 1e-6 over the 600 steps (the paper's text; the code's per-epoch stepping has no meaning at a sub-epoch scale); no gradient clipping; fp32. Trainable: `traj_proj`, `time_mlp`, `layers`, `control_head` = **9,474,832** parameters (the sampler path; `conf_head`, `feat_proj`, `cond_proj` and the trunk get no gradient). Model in `eval()` (the decoder has no dropout; the encoder's BatchNorm stays frozen).

**Scale, stated:** 600 steps × 4 windows = 2,400 windows ≈ 0.97 passes over the 2,464 eligible RL-train windows. The release trains 10 epochs of navtrain at batch 512. This is a mechanism validation at ≈ 0.4 % of the release's optimiser samples, not a reproduction of its scale.

## 4. Data, split, contamination

* Clips: the 141 B1 v7.2 EVAL clips (`C:/Users/Admin/tanitad-data/refav1-eval141/eps`), the only dev-box v2ep set with poses, actions, v7.2 labels and a verified agent join. The 192 dev-box B1-TRAIN clips are frames-only ("NO poses/actions") and cannot supply GT.
* **Split, fixed before any arm:** sort by `sha256(clip_id)`; the first 41 are HELD-OUT, the other 100 RL-TRAIN (`raw/SPLIT_eval141_sha12.json`). All 3 SAM3-map clips fell in HELD-OUT by that order.
* **Contamination statement:** the 100 RL-train clips are burned for every checkpoint this prereg produces. Those checkpoints are validation artifacts and must never be scored on the 141-clip panel or entered on a leaderboard. BASE never trained on any of the 141 (registry: B1 train split, 4,572 clips).
* Windows: the harness's global stride 5; a full 6 s future; agent labels on all 50 raw frames the proxy reads. Eligible RL-train windows: **2,464** (MEASURED; dropped: split 1,404 / future 783 / agents 172).

## 5. Metrics

### 5.1 Training-side (T0), every step, `metrics.jsonl`
`loss, rl_part, rl_coef_abs_sum, il_mean_m, il_coef, grad_norm, param_delta_norm, reward_mean, reward_best_mean, human_pdms_mean, human_nc_eq_1_frac, cand_{nc,ep,ttc,comfort}_mean, frac_positive_before_bar, frac_admitted_by_bar, frac_positive_after_bar, frac_constraint_fail, frac_nonzero_adv_used, rows_with_positive, il_weight_mean`, the clamp-binding rates, timings, `finite`.

### 5.2 Held-out T0 read — PRIMARY
`ddv2_rl_refcv5.py heldout --stride 10` on the 41 held-out clips for BASE, RL-s0, NORL-s0, RL-s1: the **deployed** sampler (2 steps, η = 0 as trained, one sample per anchor), **the same inference noise per window for every checkpoint** (`seed = 500000 + k`). Per window: `fan_pdms_mean, fan_pdms_best, fan_nc_fail_frac, fan_endpoint_spread_m, fan_minade_m, sel_pdms, sel_{nc,ttc,ep,comfort}, sel_ade_m, sel_fde_m`.
**Statistic:** paired per-window differences; **clip-cluster bootstrap**, 2,000 resamples, seed 0, 95 % percentile CI; n reported as windows and clips.
**PRIMARY ENDPOINT:** `Δfan(X) = fan_pdms_mean(RL-sX) − fan_pdms_mean(NORL-s0)`, X ∈ {0, 1}.

### 5.3 T1 read (self-action OPEN loop)
`taniteval/tools/refcv3_arm.py` (grid 2s, stride 5, infer-seed 0, `--with-oracle-sel` off) on the held-out directory for all four checkpoints; four families from each JSON; `taniteval/tools/paired_openloop.py` for RL-sX vs BASE and RL-sX vs NORL-s0 (paired episode-cluster bootstrap, 2,000). Every driving number is reported with all four families (strategic is UNAVAILABLE on this corpus, per the harness).

## 6. Criteria

**Integrity — all must hold, else the run is VOID (not negative):**
* **I1** every logged step finite in all three arms; `param_delta_norm > 0` at the end.
* **I2 (known value)** NORL: `rl_part == 0.0` and `rl_coef_abs_sum == 0.0` on every step.
* **I3** RL arms: `frac_positive_after_bar > 0` on ≥ 90 % of steps.
* **I4 (known value)** pooled over all training windows of all arms, the human's own proxy NC = 1 on ≥ 98 % (MEASURED 12/12 in D5). A lower value means the frame, ego box or agent transform is wrong.
* **I5 (known value)** two BASE held-out T0 reads with the same seeds agree **bitwise** on every per-window metric.
* **I6 (known value, where computable)** BASE T1 on the held-out roll vs the banked landing dump (`C:/Users/Admin/refcv5cmp/out/refcv5-v2_dump`, mirror code `refc.py` md5 `c912896e…`) on shared (clip, raw frame) windows: |Δ ADE| ≤ 0.005 m. If the stride phases share no window, I6 is reported **NOT RUN**, not passed.

**H-DDV2RL-1:**
* **SUCCESS** — `Δfan(0)` and `Δfan(1)` both > 0 with 95 % CI lower bounds > 0.
* **PARTIAL** — exactly one seed meets SUCCESS and the other's CI contains 0.
* **FAILURE** — both CIs contain 0, or either CI upper bound < 0.

**H-DDV2RL-2 (harm guard):** **FAIL-HARM** if, for BOTH RL seeds, T1 `ade_m` of `os` vs BASE has a CI lower bound > 0 (worse). Otherwise "no harm detected at this n" (not a claim of no harm).

**Reported without a criterion:** fan spread (DDv2 Tab. 3 predicts diversity falls); `sel_*` deltas; NORL − BASE (the IL term's own effect); fan collision fraction; the clamp-binding rates over training.

## 7. Which variance each interval answers
* The clip-cluster bootstrap answers **"would another draw of held-out clips say this?"**. It does not see training-seed or inference-noise variance.
* **Training seed:** two RL seeds, reported as a range, with no CI at n = 2. **NORL has one seed**, so `Δfan` carries NORL's own seed variance unmeasured. Stated as a limitation, not corrected for.
* **Inference noise:** the T0 read pairs the noise per window across checkpoints, so it cancels in the differences. The T1 roll uses infer-seed 0 for every arm; the registry's T1 inference-seed floor for refcv5-v2 is ≈ 0.0001 m ADE (`MODEL_REGISTRY.md`).

## 8. Budget (≤ 3.0 GPU-hours on the RTX 4060)
MEASURED in the smoke run: ≈ 2.7 s per step ⇒ ≈ 27 min per arm ⇒ **≈ 81 min** for three arms. Plus T0 reads 5 × ≈ 4 min (including I5) and T1 rolls 4 × ≈ 11 min, **≈ 64 min**. Already spent: ≈ 10 min on diagnostics and smoke. **Total ≈ 2.6 GPU-h.** If the cumulative total passes 2.75 h, the NORL T1 roll is dropped first and that is recorded.

## 9. On FAILURE — the next levers (Rule Zero), with pod cost as ESTIMATED
1. **Clamps off** (`--no-input-clamp --no-clip-sample`). D3 MEASURED the release clamps moving the chain's fan 11.8 m from the deployed fan's endpoints, against 6.1 m without them. Dev box, ≈ 27 min per arm.
2. **Put labels 18 … 2 on support.** Retrain the sampler with DiffusionDrive's random-t objective (`sampler_train_t_max` is a dead flag, `refc_sampler.py:397-404`), or run the RL chain on the deployed ladder [10, 0] as a declared deviation. The retrain is a pod job: a refcv5-v2-length run, ≈ 40 k steps (registry).
3. **Scale.** The release's 10 epochs at batch 512 over the 4,572-clip B1 train corpus. ESTIMATED from the dev box's 2.7 s per 4-window step: ≈ 0.68 s per window, ≈ 0.25 s per window on an A40-class GPU if frame decode is cached. At about 114 k windows per epoch that is ≈ 8 GPU-h per epoch and ≈ 80 GPU-h for ten, before caching the frozen trunk's features. Caching them removes the frame decode and trunk forward (≈ 55 % of the measured step), roughly halving it. This needs the pod plus a feature-cache job.

## 10. Execution record
Filled by `RESULT.md` in the package, including any deviation from this file.
