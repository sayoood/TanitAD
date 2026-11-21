# PREREG: H-ADAM -- was RL-S1's EPO dose voided by AdamW's per-parameter normalisation?

Written 2026-10-15 by the Research Lab agent (Sonnet) BEFORE any gradient, Adam update or parameter-displacement number in section 3 was computed.
Done before writing this file (none of it is a registered statistic): HF download plus sha256 check of the inputs; the key-overlap census between the trunk cache
and the RL sets (302 RL-train and 62 RL-val frames have a cached `visual_ctx`); a fwd+bwd timing probe on RANDOM tensors; reading `scorer_finetune.py` at the
remote tip `af1e2a8` (blob sha256 in REPORT.md). Nothing was read from any banked scorer state.

## 0. The hypothesis and why the suggested metric needs one correction

**H-ADAM.** AdamW divides each parameter's update by the running scale of that parameter's own gradient. Where the EPO term dominates a tensor's gradient, multiplying
EPO by 0.25 changes the gradient scale and the update direction's magnitude by almost nothing, so R0b (lam 0.25) moved the scorer about as far as R0 (lam 1.0). Banked
evidence it must explain (MEASURED, `../2026-10-15-rls1-calib/REPORT.md`, RL-val selection tier): PDMS +0.53 / +0.90 (R0b u250 / u500) against +0.60 / +0.93 (R0
u250 / u500); TTC -0.60 / -0.73 against -1.00 (R0 u1000).

**Correction to the suggested statistic (stated before measuring).** The suggested "norm ratio of the Adam update vectors for lam 0.25 vs 1.0" is a WEAK discriminator: Adam bounds
every coordinate's step by about lr, so `||D(0.25)|| / ||D(1.0)||` (D = the full displacement) is near 1 whether or not the dose acts, because BCE-driven coordinates move
too. The statistic that answers the question is the **EPO-attributable displacement** `Delta(lam) = D(lam) - D(0)` (D(0) = the same stream with EPO off). Both are
computed; the suggested one is reported as SECONDARY and the positive control C3 below demonstrates that it cannot tell the cases apart.

## 1. Fixed set-up (no degrees of freedom left after this section)

* **Optimiser, from `scorer_finetune.py` @ `af1e2a8` and R0's banked argv (`raw/2026-10-14-rls1-r0-kill`, PASS token argv sha256 `24451b62...`):** `torch.optim.AdamW(lr, weight_decay=0.01)`
  with betas (0.9, 0.999) and eps 1e-8 (torch defaults, the trainer names none); `lr` peak 1e-4 (`--lr 1e-4`); `lr_at(u) = lr*(u+1)/50` for u < 50 else a cosine to total = 1000;
  `clip_grad_norm_(paramsB, 1.0)` over arm B's whole parameter list; 8 micro-batches of 8 sets per update (`--batch 8 --accum 8`), each loss divided by 8.
  EPO: `lam_pg(u) = lam * min(1, u/200)` (`--epo-ramp 200`), `--epo-temp 0.1 --epo-beta 0.1 --epo-lam-d 0.15 --epo-w-prog 0.5`, `--lane-head`, `--lam 0 --b-mode listnet`
  (so arm B = BCE + lane BCE + EPO). R0b's argv differs from R0's in `--out` and `--epo-lam 0.25` only (checked by diffing the two banked PASS tokens).
* **Real torch `AdamW` and real `clip_grad_norm_` are used; Adam is not re-implemented.** The loss is the trainer's OWN source: module-level `epo_loss`, `epo_policy`, `epo_reward`,
  `epo_ramp`, `agg_v1` are imported; the closures inside `main()` (`lane_bce`, `sel_score`, `losses`, `losses6`, `epo_term`, `train_losses`) are extracted from the file's AST VERBATIM
  and exec'd with the R0 argv namespace. Nothing is retyped.
* **Parameters:** the three scorer modules `score_q_mlp`, `score_dec`, `score_head` (7 outputs, native lane head) loaded STRICTLY from the base checkpoint
  `model_sft4_B_u400.pt` (sha256 `8b4dba48...`, the `--ckpt` of R0). "Tensor" below = one `named_parameter` of arm B.
* **Data:** RL-train sets of `rlfan_sets_train` (sha256 `d10a8a71...`), labels from `rlfan_pdm_train`, `rlfan_terms_train`, `rlfan_tlane_train` (all sha256-checked against
  `MANIFEST_rlfan_sha256.txt`), restricted to the frames whose `visual_ctx` is in the local trunk cache. Batches of 64 sets are drawn without replacement
  within an update, with replacement across updates, seeds 20261015 / 20261016 / 20261017 (three independent streams).
* **sctx (declared substitution, not hidden).** The scorer's context `visual_ctx` [7680, 256] is read from the trunk cache `data/refe_trunk_cache/d7c59f4f`, built from
  `snap_epoch015.pt` (step 4933), NOT from the encoder R0 trained under (final model, step 10075 + SFT-4). The camera pixels are not on the dev box (80 GB, on the pod only) and a ViT-L
  forward is not a CPU job. This is the largest threat to validity; section 2 registers the test that decides whether the substitute is admissible.
* **Numerics:** CPU float32 (the trainer ran bf16 autocast on an A40). One batch is compared fp32 vs CPU-bf16-autocast (control C8).

## 2. Face validity of the substituted sctx (decided BEFORE any gradient number is read)

On the RL-val frames present in the cache (expected 62), run the BASE scorer on the cached sctx and the banked set trajectories and compare with the banked
`scores_step0_base.npz` logits (computed under the real encoder). Statistic: for each of the 7 heads, the mean over scenes of the within-scene Pearson r (64 proposals)
between the two logit vectors; V = the median over heads.

| V | status | consequence |
|---|---|---|
| >= 0.90 | ADMISSIBLE | the simulation is ESTIMATED and may decide the mechanism row of section 5 |
| 0.60 .. 0.90 | DEGRADED | the simulation is reported as ESTIMATED-low and may not decide; the verdict rests on the realised displacement (section 4) with the mechanism row marked UNTESTED |
| < 0.60 | INADMISSIBLE | the simulation is reported for transparency only; verdict from section 4 alone |

## 3. Measurements

**3.1 Gradient scale ratio (the suggested r).** At theta_0, on K = 50 batches per stream, per tensor p:
`r_p(lam) = lam * sqrt(mean_t ||g_E,p,t||^2) / sqrt(mean_t ||g_B,p,t||^2)`, with g_B the gradient of arm B's BCE + lane-BCE and g_E the gradient of the un-scaled EPO term
(`-J + beta*KL`, lam_pg = 1). Reported for lam in {0.25, 1.0}. EPO "dominates" tensor p at dose lam when r_p(lam) > 1. Also the fraction of coordinates with
`|lam g_E| > |g_B|` (rms over t). Mechanism requirement: r_p(0.25) >= 1 on the core tensors C80 (below).

**3.2 Adam simulation.** For each of the three streams, for lam in {0, 0.25, 1.0}, K = 50 real AdamW steps in two regimes:
* **T (as trained):** `lam_t = lam*min(1, t/200)`, `lr = lr_at(t)` (warm-up), t = 0..49.
* **P (post-ramp):** `lam_t = lam`, `lr = 1e-4` constant, cold Adam state (isolates the optimiser's normalisation from the ramp / warm-up transient).

Two simulation modes: **open-loop** (OL; gradient streams precomputed at theta_0 and recombined `g_t = g_B,t + lam_t g_E,t`, clipped, stepped) -- the primary, because it makes the
BCE-removed and BCE-amplified controls exact; and **closed-loop** (CL; the real trainer step, gradients recomputed at the moving theta) -- the check that open-loop is not an artefact.
`D(lam) = theta_K - theta_0`, `Delta(lam) = D(lam) - D(0)`.

Statistics (per stream, then mean over streams; the spread over streams is reported):
* **S_p = ||Delta_p(0.25)|| / ||Delta_p(1.0)||** per tensor; **S = sqrt(sum_p ||Delta_p(0.25)||^2 / sum_p ||Delta_p(1.0)||^2) over C80**, where **C80** = the tensors, taken in descending
  `||Delta_p(1.0)||^2`, whose cumulative share reaches 80 % of the total ("the tensors carrying most of the EPO update"). Dose elasticity `e = ln S / ln 0.25` (e = 1 proportional, e = 0 voided).
* cos(Delta(0.25), Delta(1.0)) over C80; SECONDARY (suggested form): `U = ||D(0.25)|| / ||D(1.0)||` and cos(D(0.25), D(1.0)) over all tensors.
* Primary reading: **S_P** (open-loop, regime P, K = 50). Also reported: S_T, the CL versions, and a K sweep {25, 50, 100} in OL regime P (deciding nothing).

**3.3 Realised displacement (REAL, no substitution).** From the banked `scorer_states_u{250,500}.pt` of R0, R0b and C0 and the base checkpoint: per tensor
`d_p(X, u) = ||theta_B(X, u) - theta_A(X, u)||` (arm B minus the SAME run's BCE-only arm A), X in {R0, R0b, C0}. C0's B arm is BCE-only too, so `d_p(C0, u)` is the run-to-run
numerical floor (CUDA non-determinism; the dumped scores of C0-A and C0-B differ in their sha256). Floor-corrected EPO displacement `e_p(X, u) = sqrt(max(d_p(X,u)^2 - d_p(C0,u)^2, 0))`.
**rho_u = sqrt(sum_p e_p(R0b,u)^2 / sum_p e_p(R0,u)^2)** over C80 of `e_p(R0, u)`, at u = 250 and 500; plus cos((theta_B-theta_A)(R0b), (theta_B-theta_A)(R0)) over C80 and the same ratio
uncorrected. (R0b has no u1000 state, so u1000 is not read.)

## 4. Controls that must read known values (else the instrument is void and the row it feeds is not read)

| id | control | must read |
|---|---|---|
| C1 | lam = 0 through the trainer's own `train_losses` (B arm) vs its arm-A (BCE-only) gradient on the same micro-batch, and the AdamW updates they produce | gradients and updates bit-identical (max abs diff 0.0) |
| C2 | BCE removed (`g_B := 0`), lam 0.25 vs 1.0, regime P and T | `Delta(0.25) = Delta(1.0)` up to eps: S in [0.97, 1.03], cos >= 0.999 (run with the real clip; if the clip engages in one dose only, also with the clip off, and say so) |
| C3 | BCE amplified (`g_B := 1000 g_B`), so EPO is a small perturbation | S in [0.20, 0.30] (proportional) while U stays within 0.02 of 1 -- proving the suggested statistic cannot discriminate |
| C4 | the whole OL pipeline run twice with the same seed | bit-identical |
| C5 | a micro-batch's gradient from the trainer's closures vs the same loss evaluated through the module-level functions only (`epo_loss(...)` + a hand-written BCE) | rel. diff <= 1e-5; proves the extracted closures are the trainer's |
| C6 | architecture identity: `load_state_dict(strict=True)` of the base scorer into the rebuilt modules | no missing / unexpected keys, shapes equal |
| C7 | the same lam-scaling law at the logit level: `d(-J)/d(logit)` from `epo_loss` at lam_pg 0.25 equals 0.25 x that at 1.0 | exact (this is what the trainer's own test pins; it shows the dose is exact BEFORE Adam) |
| C8 | fp32 vs CPU bf16-autocast gradient, one batch | per-tensor norm ratio r_p within 15 % |

## 5. Verdict rule and what each outcome implies (committed now)

Levels for a ratio X in {S, rho}: **HIGH** X >= 0.80; **LOW** X <= 0.35; **MID** otherwise. Mechanism row = S_P (OL, regime P) if section 2 says ADMISSIBLE, else UNTESTED.
Realised row = rho_500 (and rho_250 must not contradict it by more than one level).

| mechanism S_P | realised rho_500 | verdict | implication / next lever |
|---|---|---|---|
| HIGH | HIGH or MID | **H-ADAM HOLDS** | the dose lever was voided by the optimiser, not refuted as a lever. Design the dose where Adam cannot normalise it (two-optimiser split with a calibrated kappa; section 6 of the report), simulate it, draft R0c. |
| HIGH | LOW | mechanism present at init, NOT realised | by u500 BCE regained control (r fell as EPO converged) or the clip took over; the dose acted in parameter space. H-ADAM does not explain the run. Next: late-run r from the banked states (clip engagement) and function-space saturation of J. |
| MID | any | **PARTIAL** | report `e`; the lever is partly voided. Next: the dose that restores proportionality is computed from `e`, and the split optimiser is still the candidate. |
| LOW | LOW | **H-ADAM FAILS** | the dose acts at the parameter level; the SATURATION is in function space (J and the selection policy's softmax at T = 0.1), so magnitude is the wrong lever. Next lever = DIRECTION: the report ranks EPO-restricted-to-the-progress-head-rows and a reward / projection change by where the realised displacement lives (share in `score_head` vs the shared trunk). |
| LOW | HIGH | **H-ADAM FAILS as the mechanism; saturation real** | the parameters saturate although Adam does not normalise the dose at init: suspect the global clip (norm 1.0) or late-run normalisation. Next: clip engagement census on the banked scorer states / the stream, then a clip-off or per-term-clip arm. |
| UNTESTED (section 2 DEGRADED / INADMISSIBLE) | HIGH | **SUPPORTED, mechanism unmeasured** | realised displacement is dose-insensitive; the mechanism cannot be shown. The split-optimiser lever is still the cheapest discriminating experiment. |
| UNTESTED | LOW / MID | **NOT SUPPORTED** | the realised displacement scales with the dose; look elsewhere. |

A tie between the control-corrected and uncorrected rho (they differ by more than one level) is reported as UNRESOLVED, not rounded.

## 6. Evidence classes and limits (stated now)

* Sections 3.1-3.2: **ESTIMATED** (real scorer weights, real labels and trajectories, but a substituted sctx and CPU fp32; open-loop reuses theta_0 gradients).
* Section 3.3: **MEASURED** (banked parameter tensors; one training seed per arm; RL-val is a selection tier, a parameter-space read is not an endpoint).
* The estimator is the stream-to-stream spread of three independent batch streams. It answers "would another draw of BATCHES say this?" and nothing about another TRAINING seed or INFERENCE seed
  (H-ESTIM-SEED-1): the realised-displacement read has one training seed per arm, and the C0 B-vs-A floor is the only noise measure it has.
* No claim here changes g5, K3 or any bar of `eval/PREREG_RL_S1.md`.

## Amendment 1 (appended 2026-10-15 BEFORE any section 2 / 3.1 / 3.2 number was computed; section 3.3 had already been read: rho_250 = 0.52, rho_500 = 0.55, both MID)

Disclosure: `disp_real.py` (section 3.3) was run BEFORE this amendment, because it needs none of the substituted inputs. Its result (MID at both updates) is therefore known when the thresholds of
section 5 are applied to the mechanism row, but it was NOT known when section 5 was written; the table above was committed first (sha256 in `PREREG_H_ADAM.sha256`).

Diagnostics added, deciding nothing (they cross-check the substituted sctx at the LOSS level, which section 2 does at the logit level):
* **D1** the mean BCE (arm B terms, BCE + lane BCE) of the base scorer over the pool, against R0's logged `bceA` at update 25 (0.574, `sft.jsonl`, a mean over updates 1-25 that already includes 25 BCE steps);
* **D2** the mean `J` (exact expected reward under the base policy) over the pool, against R0's logged `epo_J` at update 25 (0.817);
* **D3** the pool's composition against the RL-train partition (share of sets with a TTC failure, mean number of failing proposals) so a skewed 302-frame pool is visible.
Section 3.2 gradient streams are drawn from three disjoint random partitions of the pool into micro-batches of 8 (one partition per stream).

## Amendment 2 (appended 2026-10-15 AFTER all registered measurements were read; nothing above is edited)

* **Table gap.** Section 5 does not enumerate the cell (mechanism S_P LOW, realised rho MID). Observed: S_P = 0.327 (LOW), rho_250 = 0.520 and rho_500 = 0.549 (MID). Both enumerated LOW-mechanism rows read "H-ADAM FAILS as the mechanism", so that is the verdict applied; the cell was assigned after seeing the data and is flagged as such.
* **First open-loop pass had a scale bug** (mean instead of sum of accum-divided micro-gradients: 8x too small, clip never engaged). Same conclusion (S_P50 0.316-0.333); superseded by `adam_sim.json`; the faulty file is kept as `adam_sim_WRONGSCALE_mean_not_sum.json`.
* **C2 and the registered window.** With the corrected scale C2 reads S 0.9988 (P) and 0.9875 (T), cos >= 0.9999: inside the registered window. (The faulty pass had read 0.9395 in regime T, outside it, because eps was relatively 8x larger.)
* **Post-hoc analyses, NOT registered, labelled as such in the report:** the closed-loop runs (`adam_closed.py`; the closed-loop mode itself was registered in 3.2, the draw count and the beta-0 ablation were not), the beta-0 ablation (`closed_beta0_stats.py`), the late-run gradient ratio and J/KL probe (`late_probe.py`), the A/B-versus-base decomposition of the g5 gap (`ab_vs_base.py`), and the face-validity negative control (`face_validity_swap.py`).
* **Dropped:** the split-optimiser lever simulation of the brief's step 3 (it applies only if H-ADAM holds; its simulated dose scaling would be S = kappa by construction).
