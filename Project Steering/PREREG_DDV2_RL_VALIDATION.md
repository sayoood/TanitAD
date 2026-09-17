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

---

# EXTENSION — lever **L1 / D9**, the mode-preserving imitation term + gradient clipping

**Written:** 2026-09-16, **before any L1 arm ran**; **0 GPU-minutes spent on this lever.**
**Owner:** Arch+Inference implementation/validation agent. **PI request:** *"Do what is necessary to prepare the RL post training, implement D9"* (2026-09-16), executing §9 **LEVER 1** of `…/2026-09-15-ddv2-rl-prep/RESULT.md`.
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-16-refcv6-rl-l1/` (RESULT, DIFF, code).
⛔ **§§1-10 above are the 2026-09-15 pre-registration and are NOT rewritten.** They stand as the record of the arms that already ran and FAILED. Everything below is an extension, and every clause that is carried forward unchanged says so.

**What ran BEFORE this extension, and why it does not contaminate the test:** the implementation and its 110 CPU tests (`stack/tests/test_ddv2_il.py` 37 + the 73 pre-existing), including the synthetic-fan mutation proof. **No model was trained, no checkpoint was produced and no held-out number of any L1 arm exists.**

## 11. What changes, and what does not

The measured cause of the 2026-09-15 FAILURE was the release's **imitation** term, not its RL term (§7.2-7.3 of that RESULT: fan endpoint spread **37.5199 m → 2.4457 m**, −93.5 %; NORL−BASE T1 ADE **+0.081 m [0.051, 0.116]** against RL−BASE's +0.083 m; RL−NORL **−0.0012 m [−0.0285, +0.0268]**, not separated). One seed diverged at grad norm **15,712**.

**Exactly two changes** (element-by-element proof: `…/2026-09-16-refcv6-rl-l1/DIFF_L1_VS_RELEASE.md`):

* **C1** — the all-modes L1 over all `G·N = 468` chains is replaced by the **mode-preserving matched-anchor** L1 the REF-C trainer already uses (`stack/scripts/refc_v3_train.py:2383-2391`), matched over the **decoded** bank (`inp.bank` IS `out["anchor_bank"]`, `refc.py:2615`), normalised as the trainer normalises it so the batch's **total** imitation budget is unchanged.
* **C1b** — the **plain λ ≈ 0.01** variant of the release's own form, exposed as the cheaper alternative arm **and** as C1's confound control.
* **C2** — `clip_grad_norm_(params, GRAD_CLIP)`, **once**, after all ten per-step backwards and before `opt.step()`. **`GRAD_CLIP = 100`** — see AMENDMENT A-1 immediately below.

### ⭐ AMENDMENT A-1 — the gradient-clip max-norm, **1.0 → 100**

**Amended 2026-09-16. NO ARM HAD RUN AT AMENDMENT TIME** — zero GPU-minutes had been spent on this lever, no L1 checkpoint existed and no L1 held-out number existed. This is a **pre-run amendment**, not a post-hoc one.

**Why.** The extension as first written pre-registered **1.0** (the PI's instruction) and escalated a measured problem with it. **MEASURED** on the banked logs (`…/2026-09-15-ddv2-rl-prep/raw/metrics_arm-*.jsonl`, 600 steps × 3 arms) and **INDEPENDENTLY VERIFIED by the coordinator on the same three files**:

| threshold | RL-s0 | NORL-s0 | RL-s1 (diverged) |
|---|---|---|---|
| steps with `grad_norm > 1.0` | **600/600** | **600/600** | **600/600** |
| steps with `grad_norm > 100` | **1/600** | **1/600** | **276/600** |
| minimum norm observed | 1.87 | 2.22 | 3.45 |
| median norm | 16.67 | 17.93 | 61.96 |
| maximum norm | 147.3 | 146.0 | **15,712.3** |

⇒ **1.0 is a 17-62× rescale on every single step; 100 is an actual spike guard.** C2 exists to stop the RL-s1 divergence, not to re-scale a healthy run, so the max-norm that implements C2's stated purpose is **100**.

**Coordinator's ruling, 2026-09-16, recorded verbatim in substance:** *"Your escalation is upheld and the clip threshold changes to 100 — I verified your gradient-norm numbers myself against the banked logs. 1.0 was my number and it was wrong. You were right to implement it as instructed and escalate rather than silently substitute."*

**What A-1 changes:** `ddv2_il.GRAD_CLIP` and therefore the `--grad-clip` default, from `1.0` to `100.0`; and every Stage-A / Stage-B arm command in §12 now reads `--grad-clip 100`. **`--grad-clip 1.0` and `--grad-clip 0` remain available as declared arms** — 1.0 becomes interesting **only if the diverging seed also fails at 100**, and running it would be a new arm with its own record, not a re-read of these.

**What A-1 does NOT change — nothing else in this extension is re-derived or touched:** the F1 fan-collapse gate stays **CI lower bound > −15.01 m, i.e. ≥ 60 % of the 37.5199 m cold start** · the primary endpoint `Δfan` and its statistic · the arms, split, cold start, seeds, step count and batch · I1-I10 · the budget · the scope limits. A-1 is a **single scalar**.

Under AdamW a *constant* rescale of the whole gradient largely cancels in `m/(√v+ε)` (**REASONED, not measured**) — which is why the every-step behaviour at 1.0 might have passed unnoticed — but the rescale is not constant across steps and Adam's `v` is an EMA across them, so the optimiser trajectory does change. §13.4 still reports the clip's own effect against the banked arms, and **I10** still refuses a run whose clip never bound while its peak norm exceeded the max-norm, so at 100 a dead flag cannot pass either.

**Unchanged, and unchanged by construction** (`git diff` on `stack/tanitad/rl/ddv2_rl.py`, `ddv2_refc_chain.py`, `pdm_proxy.py` is **empty**): truncated start t = 8 · the 10-label chain [18…0] with one-unit transitions and ᾱ(−1) = 1 · two-scalar exploration with its **0.04** floor and the drawn-then-zeroed additive pair · the **0.10** likelihood floor · **G = 4** · intra-anchor normalisation `(r − mean_G)/(std_G + 1e-4)` · per-sample truncation with the **≥GT** mask and the −1 constraint branch applied last · **γ = 0.8** · REINFORCE averaged over **non-zero-advantage** samples · the IL row weights' derivation from the advantage (0.1 / 1.0) · the batch-global 0-dim IL scalar · **AdamW 2e-4 / wd 1e-4** with 10 % linear warmup + cosine to 1e-6 · both release clamps ON · η = 1 · the NORL control's definition · the reward proxy, corpus, split, window filter and the deployed held-out read.

The only edited tracked file is the driver `stack/scripts/ddv2_rl_refcv5.py` (**+73 / −11**). `--il-form release --grad-clip 0` reproduces the 2026-09-15 recipe exactly, and `run.json` records `il.is_release` on every run.

## 12. Arms (same split, same cold start, same statistic)

Cold start `C:/Users/Admin/refcv5v2_final/ckpt.pt`, md5 `9405ec73b2d797c4cebd44f82dbce54b`, step 40,284 (the driver refuses another). Split: **the same** `raw/SPLIT_eval141_sha12.json` — 41 held-out clips, 100 RL-train clips. All arms `--steps 600 --batch 4`, release defaults otherwise.

### Stage A — primary

| arm | command | differs from **L1-RL-s0** by |
|---|---|---|
| **BASE** | the cold start, untrained | no training |
| **L1-RL-s0** | `train --arm rl --seed 0 --il-form matched --grad-clip 100` | — |
| **L1-NORL-s0** | `train --arm norl --seed 0 --il-form matched --grad-clip 100` | the policy-gradient coefficient is exactly 0 on every step; the IL weights stay the release's advantage-derived 0.1 / 1.0 (the §3 reasoning is carried forward unchanged) |
| **L1-RL-s1** | `train --arm rl --seed 1 --il-form matched --grad-clip 100` | window order and chain noise |

### Stage B — the cheaper alternative arm, CONDITIONAL

| arm | command | what it answers |
|---|---|---|
| **L1λ-NORL-s0** | `train --arm norl --seed 0 --il-form lambda --grad-clip 100` | the release's **form** at **λ ≈ 0.01**, IL only: *was the fan saved by mode preservation, or merely by less imitation pressure?* |

**Run condition:** Stage B runs **only if** the cumulative GPU time after Stage A is ≤ **2.45 h**; otherwise it is recorded **NOT RUN**, never "unnecessary".

**Held-out reads:** `heldout --stride 10` on the 41 held-out clips for BASE, BASE-repeat (**I5**) and every trained arm — deployed sampler, **identical inference noise per window across checkpoints** (`seed = 500000 + k`). T1 rolls (`refcv3_arm.py`, grid 2 s, stride 5, infer-seed 0, all four metric families) for BASE and the three Stage-A arms.

## 13. Criteria

### 13.1 F1 — THE GATE: the fan must not collapse

`Δspread(X) = fan_endpoint_spread_m(X) − fan_endpoint_spread_m(BASE)`, **paired per window** with identical inference noise, **clip-cluster bootstrap** (2,000 resamples, seed 0, 95 % percentile CI), on the same held-out read.

> **F1 PASS** for arm X iff the 95 % CI **lower** bound of `Δspread(X)` is **> −15.01 m** — X retains **≥ 60 %** of the cold start's **37.5199 m** (i.e. mean spread ≥ **22.51 m**).
> **F1 FAIL** iff the CI **upper** bound lies below that line. Otherwise **F1 INCONCLUSIVE**.

The threshold, stated against the measured reference points: the **cold start** is 37.5199 m (100 %); **the measured failure** — the release's IL term alone — is **2.4457 m (6.52 %)**, which fails the bar by 53 pp; the **diverged seed** RL-s1 sat at 19.6838 m (52.46 %) and **also fails**, so a bar this programme's own known-harmful arm could pass is excluded; DDv2's own reported diversity drop (P Tab. 3, −28 % ⇒ ≈ 72 % retained) **passes**, so the bar does not forbid the sharpening the paper claims is desirable.

**F1 is read BEFORE the primary endpoint.** If the fan collapses again, the lever is refuted and `Δfan` is reported but decides nothing.

### 13.2 PRIMARY ENDPOINT — `Δfan` must separate

`Δfan(X) = fan_pdms_mean(L1-RL-sX) − fan_pdms_mean(L1-NORL-s0)`, X ∈ {0, 1} — the **same** endpoint, pairing and statistic as §5.2.

> **SUCCESS** — L1-RL-s0, L1-NORL-s0 and L1-RL-s1 all pass **F1**, AND `Δfan(0)` and `Δfan(1)` both have 95 % CI lower bounds **> 0**.
> **PARTIAL** — F1 passes for all three, exactly one seed separates, the other's CI contains 0.
> **FAILURE** — F1 fails for any trained arm, OR both `Δfan` CIs contain 0, OR either upper bound < 0.

⚠️ **PRE-REGISTERED AS INVALID:** comparing an L1 arm's `fan_pdms_mean` with a 2026-09-15 arm's as if higher were better. Collapse RAISES that mean by making the 117 members alike (MEASURED NORL−BASE **+0.2205 [0.1836, 0.2562]** while the fan's best member and its human coverage got worse). `fan_pdms_mean` is compared only within this package, against a control of the same spread, and always reported beside the spread, the oracle and the best-of-117 ADE.

### 13.3 H-DDV2RL-2 — the harm guard, carried forward unchanged

**FAIL-HARM** if, for **both** RL seeds, T1 `ade_m` of `os` vs BASE has a paired CI lower bound > 0. Otherwise "no harm detected at this n" (not a claim of no harm). If L1-RL-s1's T1 roll is dropped for budget, the guard is reported **unevaluable**, never passed.

### 13.4 Reported with a direction but NO criterion

Fan **oracle** proxy and **best-of-117 ADE** (the metrics the collapse actually hurt: MEASURED 0.994 → 0.956 and 0.888 → 1.532 m) · `sel_*` deltas · NORL − BASE under the new form · fan collision fraction · clamp-binding rates · the within-run canaries `chain_endpoint_spread_m`, `il_all_modes_m` vs `il_matched_anchor_m`, `match_n_distinct_anchors_matched`, `grad_clipped` · and, as mechanism evidence for "RL now has something to rank", `frac_positive_after_bar` and the within-group reward spread against the 2026-09-15 values (RL-s0 4.2 % → 6.4 %).

**The clip's own effect** (per the escalation in §11): the fraction of steps on which the clip bound (**expected ≈ 0.2 % on a stable arm** at the amended max-norm of 100, against ≈ 100 % had A-1 not been made), the pre-clip `grad_norm` distribution, and `param_delta_norm` against the banked arms' step-600 values (RL-s0 **7.81**, NORL-s0 **8.56**, RL-s1 **11.03**). A value far below those would say the clip changed the optimiser's trajectory materially — which the AdamW argument predicts it will not, and which this diagnostic exists to check rather than assume.

### 13.5 Integrity — all must hold, else the run is VOID (not negative)

**Carried forward unchanged from §6:** I1, I2 (the D-1 amended form), I3, I4, I5, I6.

**New, and each proven able to FAIL by mutation** (`…/2026-09-16-refcv6-rl-l1/code/check_arm_l1.py`; `stack/tests/test_ddv2_il.py::test_MUTATION_each_L1_integrity_check_fires_on_its_own_defect`):

* **I7 — the lever is on, and is the one claimed.** `run.json`'s `il.form` / `il.lambda_scale` / `grad_clip` match the flags, and **every** logged step agrees with them. (The defect: an arm that declares `matched` and ran the release.)
* **I8 — the matched term is the one that drove.** Matched arms: `il_mean_m == il_matched_anchor_m` and the two IL statistics are not identical on every step. Release/λ arms: `il_mean_m == il_all_modes_m`.
* **I9 — the match is not degenerate.** Not every batch matched a single anchor on every step.
* **I10 — clipping is real and is not everything.** No step's post-clip norm exceeds the max-norm; and either the clip bound on some step, or the run's peak pre-clip norm stayed under it.

## 14. Budget — ≤ 3.0 GPU-hours on the RTX 4060

**MEASURED** from the banked logs' own wall clock: **2.60 / 2.49 / 2.46 s per step** (§8 above budgeted 2.7 s from the smoke; the completed runs came in below it). Plus L1's ≈ +2 % (two extra `no_grad` L1 evaluations, one 117×117 `cdist`, one `argmin`) ⇒ **ESTIMATED ≈ 2.65 s**. The totals below keep the conservative **2.75 s**, so they are an upper bound.

Stage A = 3 arms × 27.5 min (**82.5**) + 5 held-out T0 reads × 4 min (**20**) + 4 T1 rolls × 11 min (**44**) = **146.5 min = 2.44 GPU-h**. Stage B = **0.53 GPU-h**. **Total ≈ 2.97 GPU-h.**

**Drop order** if the cumulative passes **2.85 h**: (1) Stage B; (2) L1-NORL-s0's T1 roll; (3) L1-RL-s1's T1 roll — each recorded **NOT RUN** with the criterion it disables named. Dropping (3) makes H-DDV2RL-2 unevaluable.

**Spent so far on L1: 0 GPU-minutes.** ⛔ The dev box's RTX 4060 belongs to another agent's proof package; these arms wait for the Master Mind's slot and the GPU is checked free before each one.

## 15. Scope limits that carry forward and must appear in every L1 report

1. The three 2026-09-15 checkpoints are **BURNED** (trained on 100 of the 141 EVAL clips) and **must never be scored on the 141-clip panel**. ⛔ **The L1 checkpoints are burned on the same split and inherit the same prohibition.**
2. The dev-box scale is **≈ 0.4 % of the paper's optimiser samples**; L1 does not change that.
3. The reward is a **proxy with NO drivable-area term** — our maps do not yet cover the training clips, so DAC ≡ 1 and the release's DAC constraint branch is inert. **Never quote any number here as PDMS.**
4. T1 is **self-action OPEN loop** (PI ruling 2026-09-02); nothing here is a closed-loop claim.
5. Every UNVERIFIED item of §10 of the 2026-09-15 RESULT carries forward, and this extension adds three of its own (`…/2026-09-16-refcv6-rl-l1/RESULT.md` §8): the outcome itself, whether the matched form's concentrated per-chain budget is the right scale on the real model, and whether a preserved (across-anchor) fan actually gives GRPO the (within-anchor) variation it ranks on.

## 16. Execution record

Filled by `…/2026-09-16-refcv6-rl-l1/RESULT.md`, including any deviation from this extension. ⛔ **Any deviation is declared BEFORE the held-out number it could affect is read**, in the form §7's D-1 used.

---

## 16.1 EXECUTION RECORD — Stage A, filled 2026-09-17

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-refcv6-rl-stage-a/`.

### What ran

| | status | wall |
|---|---|---|
| **L1-RL-s0** | ✅ 600/600, `check_arm_l1.py` exit 0, `problems: []` | 1,875.5 s |
| **L1-NORL-s0** | ✅ 600/600, exit 0, `problems: []` | 2,056.8 s |
| **L1-RL-s1** | ⛔ **failed twice on CUDA OOM** under host-RAM contention (died at 157 and 186 rows), then ✅ relaunched **unchanged** on a quiet box | 658 s + ~780 s wasted, then ~1,990 s |
| held-out T0 reads | ✅ for all three trained arms (BASE and BASE-repeat were already banked 2026-09-15 on the **identical** 493 windows) | ~14 min |
| **Stage B (`L1λ-NORL-s0`)** | ⛔ **NOT RUN** — §14 drop order item **(1)** | — |
| **T1 rolls** | ⛔ **NOT RUN in Stage A** — see the deviation below | — |

⛔ **The batch was NOT reduced to get the replicate through.** Batch is held constant across arms, so
a smaller-batch arm 3 would have been a **third condition wearing a replicate's name**.

### §13.1 F1 GATE — evaluated on the pre-registered statistic

⭐ **Cold-start check is exact:** BASE mean `fan_endpoint_spread_m` reads **37.5199 m**, matching
§13.1's literal to four decimals.

| arm | Δspread vs BASE | 95 % CI | mean | retained | **F1** |
|---|---|---|---|---|---|
| L1-RL-s0 | −8.9872 | [−11.3541, −6.5142] | 28.53 m | **76.0 %** | **PASS** |
| L1-NORL-s0 | −2.1583 | [−3.9559, −0.3553] | 35.36 m | **94.2 %** | **PASS** |

⭐⭐ **This is the gate the 2026-09-15 release arm failed by 53 pp** (2.4457 m = 6.52 %). The
matched-anchor IL form plus **Amendment A-1** fixed the fan collapse, read **before** the primary
endpoint as §13.1 requires.

### ⛔⛔ DEVIATION D-2 — THE T1 ROLLS WERE NOT RUN, AND THIS DECLARATION IS **LATE**

§12 requires T1 rolls for BASE and all three Stage-A arms. **None ran.** ⇒ **`H-DDV2RL-2` is
UNEVALUABLE**, which §13.3 requires be reported as such and **never as passed**.

⛔ **§16 says a deviation is declared BEFORE the held-out number it could affect is read. The T0
numbers were already read when I noticed. This declaration is therefore LATE, and I am recording
that rather than backdating it.** ⚠️ Mitigating, and stated as mitigation rather than excuse: T1 and
T0 are different tiers on different instruments, so the late declaration cannot have been shaped by
a T0 result — but the rule exists precisely so that argument never has to be trusted.

⭐ **Why this is the package's most serious gap, not a bookkeeping nit.** 2026-09-15 read
**FAIL-HARM**: both RL seeds worse than the cold start on T1 ADE (**+0.083 [0.056, 0.115]** and
**+0.081 [0.051, 0.116]**). **Repairing that harm is what L1 exists for.** Every number in the Stage
A package is **T0** — a diagnostic tier — so without T1 the package cannot say whether L1 fixed the
thing it was built to fix.

### The remedy, and it follows the pre-registered drop order rather than my judgement

§14's order at >2.85 h is: **(1) Stage B, (2) L1-NORL-s0's T1 roll, (3) L1-RL-s1's T1 roll.**

* **(1)** already NOT RUN.
* **(2) L1-NORL-s0's T1 roll is DROPPED.** It feeds only §13.4's *"NORL − BASE under the new form"*,
  which is **reported with a direction but NO criterion**.
* **(3) L1-RL-s1's roll is KEPT** — `H-DDV2RL-2` needs **both** RL seeds against BASE.

⇒ **dropping exactly item (2) makes the harm guard EVALUABLE inside the budget:** 3 rolls ×
≈9.6 min ≈ 29 min, cumulative ≈ **2.75 h < 2.85 h**. Script: the package's `code/t1_rolls.sh`, whose
invocation is copied verbatim from the 2026-09-15 `run_validation.sh` so the two packages' T1
numbers stay comparable.

### Budget

Training 1,875.5 + 2,056.8 + ≈1,990 s, plus ≈1,438 s lost to the two OOM attempts, plus ≈14 min of
held-out reads ⇒ ≈ **2.27 GPU-h** before T1; ≈ **2.75 GPU-h** with the three T1 rolls. ⛔ **Stage B
stays NOT RUN even though ≤ 2.45 h was momentarily satisfiable** — it was satisfiable only because
the T1 rolls had not run, and §14 puts Stage B **first** in the drop order. Spending a budget freed
by skipping a required measurement on an optional arm would invert the pre-registration.

### §13.2 primary endpoint

`Δfan(s0)` = **+0.0363 [+0.0243, +0.0475]**, lower bound > 0. `Δfan(s1)` follows the replicate.
⚠️ **And §13.2's two-seed requirement turns out to be the only thing standing between this endpoint
and noise:** a run-to-run replicate on the sibling rig moves `fan_pdms_mean` by **+0.2474**, **6.8×**
the +0.0363 being used to decide. See `D-DDV2RL-SEED-FLOOR-DWARFS-THE-LEVER` and
`D-DDV2RL-LAUNCH-NONDETERMINISM` in `GOALS_AND_CLAIMS.md`.

<!-- PREREG-DDV2RL-EXEC-RECORD-STAGE-A-2026-09-17 -->
