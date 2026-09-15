# RESULT — DiffusionDriveV2's RL stage on refcv5-v2: verified, prepared, pre-registered, and validated at dev-box scale

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-ddv2-rl-prep/` · **Date:** 2026-09-15 (runs 20:18–23:03 UTC) · **Box:** dev box, RTX 4060 · **GPU spent: ≈ 2.3 h of the ≤ 3.0 h pre-registered budget.**
**PI request:** *"Prepare and validate the RL approach as described in the paper"* (DiffusionDriveV2), dev box for preparation, a pod later for heavy work.
**Companion files:** `SPEC_DDV2_RL_PAPER.md` (what the method is) · `DIFF_OURS_VS_SPEC.md` (39 components, ours vs theirs) · `Project Steering/PREREG_DDV2_RL_VALIDATION.md` (written and staged before the arms) · `raw/` (every number below).

---

## 1. Headline

**The machinery is done and proven; the recipe, run faithfully at dev-box scale, does not work on refcv5-v2 — and the reason is measured, not guessed.**

1. **PREPARED AND PROVEN.** The released RL stage is ported bit-for-bit (step, chain, advantage, loss), bound to refcv5-v2's own sampler (**parity bitwise on 12/12 real windows**), and given a PDMS-shaped reward from our joins. **73 tests green; 31 of 31 mutants killed.**
2. **VALIDATED, AND THE PRE-REGISTERED ANSWER IS NEGATIVE.**
   * **H-DDV2RL-1 (primary, T0 held-out): FAILURE.** RL vs the length-matched control: seed 0 **+0.0056 [−0.0, +0.0123]** (not separated), seed 1 **−0.2418 [−0.2844, −0.1983]** (separated, worse).
   * **H-DDV2RL-2 (T1 harm guard): FAIL-HARM.** Both RL seeds are worse than the cold start on T1 ADE.
3. **THE HARM IS THE RELEASE'S IMITATION TERM, NOT ITS RL TERM.** Removing only the policy gradient reproduces the damage: NORL vs BASE T1 ADE **+0.081 m [0.051, 0.116]** against RL-s0's **+0.083 m [0.056, 0.115]**, and RL-s0 vs NORL is **−0.0012 m [−0.0285, +0.0268]**, not separated. DDv2's IL term pulls all 468 chains at all 10 steps to the single logged trajectory and **collapses refcv5-v2's 117-anchor fan by 93 %** (endpoint spread 37.5 m → 2.45 m).
4. **ONE OF TWO RL SEEDS DIVERGED** under the release's no-gradient-clipping recipe at batch 4 (gradient norm to 15,712; that checkpoint is far worse everywhere).
5. **NOTHING HERE SAYS THE PAPER IS WRONG.** It says this recipe **at 0.4 % of the release's optimiser samples, on a control-space sampler whose labels 18…2 are off its training support, with a proxy reward and no drivable-area term**, does not transfer. The next lever is named in §9 and is a **dev-box** experiment, not a pod spend.

### Integration escalations (these need someone else's action)

| # | what | who |
|---|---|---|
| **E1** | **Nothing is committed.** **33 files are STAGED** in worktree `tanitad-wt-rl-ddv2` (detached at `d4e8bc0`), including three new `stack/tanitad/rl/` modules, a new `stack/scripts/` driver, three test files, the PREREG and this package. They need a commit and a merge decision. | repo owner |
| **E2** | **The PI's spend decision** on the DDv2 RL stage: §9 recommends a **dev-box** lever first (≈ 3 GPU-h) and puts the release-scale pod run at **≈ 39–75 GPU-h** (ESTIMATED), conditional on that lever separating. | PI |
| **E3** | **PI queue item 7, Q2** (is `v0` REQUIRED in the shared contract for July-style unconditioned builds) stays open; it changes LIB, which this task was not allowed to touch. Item 8 is closed and does not apply here. | `refc.py` owner |
| **E4** | **DAC cannot be trained on** until the SAM3 map GT exists for the training corpus on this box (3 clips today, all held-out). Production is on Thor / HF, outside this task's rules. | Data Engineering |
| **E5** | **Three validation checkpoints (1.3 GB each) exist only on the dev box** and are **burned** (they trained on 100 of the 141 EVAL clips). They must never be scored on the 141-clip panel or entered in the registry. | whoever prunes the box |

---

## 2. What the method is — SPEC verified and completed (Task 1)

The SPEC was re-verified line by line against the paper (§4.1–4.5, §5.1–5.2, suppl. §7, Tab. 7) and the released code (`rl.py` in full, plus `agent.py`, `cfg.py`, the YAMLs, `scheduler.py`, `train_eval.md`, the selector's and v1's scheduler sites). **No paper cite and no code cite was wrong.** Two omissions were fixed and five framework unknowns resolved:

| # | what | class |
|---|---|---|
| **A18 (NEW)** | **x̂0 is clamped to [−1, 1] inside every DDIM step** — diffusers' `clip_sample=True` default, because the release's three constructors pass no `clip_sample=`. So every trajectory DDv2 emits or ranks is capped at 50 m forward / 20 m lateral, and a clamped coordinate gets **zero** policy gradient at the final step, which carries 61 % of the weight. Missed by the first SPEC pass **and** by the WIP port; caught by the bitwise test. | PUBLISHED-CODE + MEASURED on diffusers 0.40.0 |
| A2 (completed) | the two `nn.MultiheadAttention` dropout sites are inert (`tf_dropout = 0.0`); the trunk's 0.1 dropouts cannot create a rollout-vs-grad-pass mismatch (the trunk runs once per batch) | PUBLISHED-CODE |
| A10 (resolved) | EP's pairwise reference is **PDM-Closed's** trajectory, not the human — the fork's own comment mislabels it "GT" | PUBLISHED-CODE-UNBANKED |
| A9 (resolved) | `MultiMetricIndex` has **two** members ⇒ the reward's gate is exactly NC × DAC, as in P Eq. 14 | PUBLISHED-CODE-UNBANKED |
| NC/TTC/C/DAC/EP | value sets, at-fault typing, the 1 s TTC projection, the six comfort thresholds, the any-tick DAC rule, centreline progress | PUBLISHED-CODE-UNBANKED |

## 3. The diff (Task 2)

`DIFF_OURS_VS_SPEC.md` covers 39 components in five groups, with a class and a citation each. In brief:

* **FAITHFUL (pinned bitwise or by literals):** truncated start, the 10-label chain, two-scalar exploration with the 0.04 floor, the 0.1 likelihood floor, the additive term × 0, G = 4, intra-anchor normalisation, per-sample truncation, the ≥GT mask, the −1 branch, γ = 0.8, REINFORCE over non-zero samples, the IL form and its λ schedule, AdamW 2e-4 / 1e-4, the PDMS aggregation.
* **FAITHFUL arithmetic, DIFFERENT effect:** both release clamps act on (a_lon/4, a_lat/3) here, not on (x/50, y/20); multiplicative exploration barely explores near-zero controls.
* **DIFFERENT by model:** labels 18…2 are off-support; 117 control anchors; the ranking head shares the trained layers (DDv2 retrains a separate selector).
* **NOT-IMPLEMENTABLE-ON-OUR-DATA:** DAC in training; EP's PDM-Closed reference and lane centreline; PDMS as a benchmark.
* **MISSING in the pre-existing library (all present in the port):** the chain, per-sample advantage, γ, any active IL term, weight decay, a schedule.

**PI DECISION QUEUE.** Item 8: closed, and **does not apply** to refcv5-v2 (strict load, 0 missing / 0 unexpected, `anchor_controls` populated). Item 7: Q1 needs no decision for this stage (neither adapter is on its path); **Q2 — v0 is required for v0-conditioned builds**, supplied on every window here and pinned by a wrong-speed regression; the shared-contract question stays with the owner (E3).

## 4. What was prepared (Task 3)

| new file (no existing file was modified — MEASURED by `git status`) | what | tests |
|---|---|---|
| `stack/tanitad/rl/ddv2_rl.py` | released step (incl. A18), advantage, loss, chain rollout, grad pass, exact per-step loss decomposition | `stack/tests/test_ddv2_rl.py` — **38** |
| `stack/tanitad/rl/ddv2_refc_chain.py` | refcv5-v2 binding: sampler-input capture, `x0_fn`, native-ladder parity instrument, clamp rates | `stack/tests/test_ddv2_refc_chain.py` — **10** |
| `stack/tanitad/rl/pdm_proxy.py` | PDMS-shaped reward from obstacle.offline joins and ego poses, plus a SAM3-map DAC mechanism | `stack/tests/test_pdm_proxy.py` — **25** |
| `stack/scripts/ddv2_rl_refcv5.py` | driver: `diagnose` / `train` / `heldout`, on the T1 harness's own loader and conditioning | the diagnostics, the smoke and the run |

* **73 tests GREEN.** The one test that reads the bank SKIPS in a sparse worktree rather than passing (`TANITAD_BANK_ROOT` points it at a full checkout).
* **Mutation proof (`raw/MUTATION_SWEEP.json`): 31 of 31 KILLED, 0 survived.** Each mutant is one plausible port error swapped into the real module at import. The first sweep left **3 survivors** — agent tokens reaching the capture, t0 overlaps not ignored, the TTC cone dropped — and a killing test was written for each before the sweep was re-run.
* **Two latent defects in the WIP tests were found and fixed:** an advantage literal that contradicted its own comment, and a regression test that would have passed for the wrong reason once the clamp landed.

## 5. Zero-training diagnostics, before the arms (T0, `raw/ddv2_diagnose.json`, 12 RL-train windows)

| id | result |
|---|---|
| **D1 parity** | the binding reproduces the deployed sampler **bitwise on 12/12 windows**, and `traj == fan[sel_idx]` 12/12 |
| **D2 clamps** | anchors 0 % outside the box; the deployed x̂0 outside on **6.6 % (a_lon) / 18.6 % (a_lat)** of coordinates; the release chain's final x̂0 on **9.2 % / 25.7 %** |
| **D3 chain vs deployed** (η = 0, same start) | fan endpoints **11.8 m** from the deployed fan with both clamps, **6.1 m** without; best-of-117 ADE and fan proxy essentially unchanged (0.88/0.86/0.89 m; 0.677/0.677/0.689) |
| **D4 cost** | B = 4: 1.61 s + 0.52 s capture, 1.5 GB peak |
| **D5 known values** | human NC = 1 on 12/12; identity control exact 12/12 |
| **D6 bar** | 9.4–11.4 % of chain samples reach the human's proxy; 3.0–3.7 % carry positive advantage |

**Smoke (4 steps, machinery only):** the release's all-modes IL L1 is **6.4 m** at the cold start and both arms collapse toward the GT within 4 steps ⇒ only a control that KEEPS the IL term can attribute anything to RL. That is why NORL keeps the release's advantage-derived λ and zeroes only the policy-gradient coefficient.

## 6. Pre-registration (Task 4)

`Project Steering/PREREG_DDV2_RL_VALIDATION.md`, **staged 20:49:16Z, first arm launched 20:50:11Z.** Arms: BASE, RL-s0, NORL-s0, RL-s1 (replicate), 600 steps × 4 windows, release constants, both clamps ON. Split: the 141 EVAL clips ordered by `sha256(clip_id)` — 41 held-out, 100 RL-train (2,464 eligible windows), the 100 burned for these checkpoints. Primary endpoint: Δfan = deployed-fan proxy reward, RL−NORL, paired per window with identical inference noise, clip-cluster bootstrap. Criteria, integrity checks I1–I6 and the budget are in that file.

## 7. Validation (Task 5)

### 7.0 Execution record and the one deviation

| stage | UTC | outcome |
|---|---|---|
| PREREG staged | 20:49:16 | — |
| RL-s0 | 20:50:11 → 21:16:31 | check PASS |
| NORL-s0 | 21:16:31 → 21:42 | **check FAILED as pre-registered (I2); the chain stopped itself** |
| D-1 declared, re-checked, resumed | 21:43:09 | PASS |
| RL-s1 | 21:43:09 → 22:07:56 | check PASS (finite) |
| 5 held-out T0 reads | 22:07:56 → 22:24:07 | 493 windows each |
| 4 T1 rolls | 22:24:07 → 23:02:34 | 1,402 windows / 41 episodes each |

**DEVIATION D-1 — the pre-registered I2 check was mis-specified; amended before any held-out number existed.**
* I2 required the logged `rl_part == 0.0` on every NORL step. MEASURED: non-zero on 600/600 steps, max **4.2e-8**, ≤ **3.0e-8 of the loss** (RL-s0's median is 0.106, seven orders larger).
* Cause: `rl_part` is LOGGED as `float(li) − float(il_coef)·float(il_i)` — a float64 subtraction from a float32 sum, never exactly zero.
* What the control actually did: the coefficient multiplying the policy gradient read **exactly 0.0 on 600/600 steps**.
* Amendment: coefficient exactly 0.0 **and** |rl_part| ≤ 1e-6·max(1, |loss|). Recorded in `code/check_arm.py`; resumed by `code/run_validation_resume.sh`; RL-s0 and NORL-s0 were not re-run.
* Added afterwards so the claim does not rest on a log line: `test_zeroed_policy_coefficient_gives_exactly_the_il_only_gradient` pins it at the gradient — a zeroed coefficient gives a **bit-identical** gradient to the IL-only pass, and a different one from the RL pass.

### 7.1 Training side (T0, `raw/metrics_arm-*.jsonl`, first 50 → last 50 steps)

| arm | chain proxy reward | IL L1 (m) | positive adv. | constraint fail | grad norm (last 50) | Δ-param | wall |
|---|---|---|---|---|---|---|---|
| RL-s0 | 0.718 → 0.894 | 3.95 → 1.33 | 4.2 % → 6.4 % | 16.6 % → 6.2 % | 14.3 | 7.81 | 1,561 s |
| NORL-s0 | 0.711 → 0.900 | 3.66 → 1.31 | (computed, unused) | tracks RL-s0 | 16.2 | 8.56 | 1,491 s |
| **RL-s1** | 0.721 → **0.669** | 3.94 → **3.40** | ≈ 4–5 % | → **21 %** | **1,394** (max 15,712) | 11.03 | 1,478 s |

* **Integrity:** I1 PASS (all arms finite, parameters moved); I2 FAILED-as-written → PASS amended; I3 PASS (RL-s0 96.8 %, RL-s1 92.8 % of steps carry positive advantage); **I4 (known value) PASS — the human's own proxy NC = 1 on 100 % of all training windows of all arms.**
* **RL-s1 diverged (MEASURED).** Isolated gradient spikes at steps 240–244 (105, 335) while clamp binding was still < 1 %; sustained from ≈ step 297; the **longitudinal** final-x̂0 clamp then climbs (7.5 % → 25 % by step 320) and the lateral one reaches **69 %** by step 400. RL-s0 and NORL-s0 each had exactly one step above gradient norm 100. **Mechanism UNVERIFIED:** the ordering fits unclipped batch-4 REINFORCE variance as the trigger (the release clips nothing and runs batch 512, ≈ 128× lower per-step variance); A18's negative ε̂-path gradient on out-of-box coordinates is a candidate sustainer.

### 7.2 Held-out T0 — the PRIMARY endpoint (`raw/heldout_*.json`)

**n = 493 windows over 40 of the 41 held-out clips** (one clip had no window with both a full 6 s future and agent labels over the scored horizon). Deployed sampler, **identical inference noise per window across checkpoints**, proxy reward. Intervals are **clip-cluster bootstraps** (2,000, seed 0, 95 %).

* **I5 (known value) PASS:** the repeated BASE read is **bitwise identical**, 0 mismatches over 493 × 13 metrics.
* **Pairing control:** the human's proxy score is identical in all five reads.

| metric | BASE | RL-s0 | NORL-s0 | RL-s1 |
|---|---|---|---|---|
| fan proxy, mean of 117 | 0.649 [0.593, 0.698] | 0.875 [0.818, 0.924] | 0.870 [0.808, 0.920] | 0.628 [0.556, 0.692] |
| fan proxy, best of 117 (**ORACLE**) | 0.994 [0.987, 0.999] | 0.956 [0.928, 0.978] | 0.957 [0.930, 0.979] | 0.844 [0.792, 0.887] |
| fan NC-fail fraction | 0.168 [0.122, 0.221] | 0.082 [0.043, 0.128] | 0.086 [0.044, 0.135] | 0.195 [0.128, 0.270] |
| **fan endpoint spread (m)** | **37.5** [32.8, 42.0] | **2.46** [2.06, 2.90] | **2.45** [1.95, 2.99] | 19.7 [17.6, 21.9] |
| fan best-of-117 ADE (m, to 6 s) | 0.888 [0.779, 1.001] | 1.532 [1.230, 1.864] | 1.543 [1.230, 1.896] | 3.664 [3.184, 4.196] |
| selected-plan proxy | 0.908 [0.865, 0.946] | 0.872 [0.809, 0.923] | 0.895 [0.848, 0.937] | 0.629 [0.552, 0.698] |

| paired difference | fan proxy mean | selected proxy | notes |
|---|---|---|---|
| **RL-s0 − NORL-s0 (PRIMARY, seed 0)** | **+0.0056 [−0.0, +0.0123]** not separated | −0.024 [−0.057, +0.008] | only sel comfort separates (+0.012) |
| **RL-s1 − NORL-s0 (PRIMARY, seed 1)** | **−0.2418 [−0.2844, −0.1983]** separated | −0.267 [−0.325, −0.212] | every metric separated, worse |
| NORL-s0 − BASE (**the IL term alone**) | +0.2205 [0.1836, 0.2562] | −0.013 [−0.049, +0.021] | **spread −35.1 m**, oracle −0.037, best-of-117 ADE +0.66 m |
| RL-s0 − BASE | +0.2261 [0.1906, 0.2617] | −0.036 [−0.080, +0.001] | sel TTC −0.049 [−0.100, −0.008] |

⇒ **H-DDV2RL-1 = FAILURE** by the pre-registered rule (seed 1's CI lies entirely below 0; seed 0's contains 0). The verdict is decided on `ci.py`'s unrounded `separated` flag: seed 0's printed "−0.0" lower bound is a negative number rounded for display.

**Reading, in order of confidence.** (1) The release's IL term **collapses the fan by 93 %** and raises the fan's *mean* only by making all 117 members alike; the fan's best member and its coverage of the human get worse. (2) With stable training the RL term adds **nothing measurable** to the fan. (3) The diverged seed is far worse everywhere.

**The null is not merely a deployment-transfer artifact.** This endpoint scores the DEPLOYED sampler (2 steps, η = 0) while RL optimises the 10-step chain, so a chain-only gain could in principle hide here. It does not: on the training side, where the chain itself is scored, the two arms are indistinguishable over the whole run — chain proxy reward per 100-step bin, RL-s0 vs NORL-s0: 0.779/0.780, 0.830/0.848, 0.840/0.842, 0.875/0.878, 0.876/0.876, 0.877/0.882. **RL did not beat its control on its own objective's proxy either.** A held-out read of the 10-step chain was not pre-registered and was not run.

### 7.3 T1 — self-action OPEN loop (never "closed loop"), four families

`refcv3_arm.py` on the 41 held-out clips, **n = 1,402 windows / 41 episodes**, 2 s grid, infer-seed 0; paired comparisons by `paired_openloop.py` over the shared `ha0` floor (paired **episode-cluster** bootstrap, 2,000).

| arm | ADE (m) | FDE (m) | LON speed MAE (m/s) | LON along MAE (m) | LAT cross MAE (m) | LAT heading MAE (deg) |
|---|---|---|---|---|---|---|
| BASE | **0.2994** [0.246, 0.370] | 0.630 | 0.264 | 0.2595 | 0.0908 | 2.37 |
| RL-s0 | 0.3820 [0.327, 0.451] | 0.775 | 0.328 | 0.3061 | 0.1577 | 2.78 |
| NORL-s0 | 0.3808 [0.326, 0.443] | 0.745 | 0.319 | 0.2969 | 0.1593 | 2.43 |
| RL-s1 | 0.9733 [0.854, 1.096] | 1.272 | 0.517 | 0.4470 | 0.7254 | 8.38 |

*(BASE on this subset agrees with the registry's full-141 value for the same checkpoint, 0.3079 [0.2795, 0.3390].)*

| paired (margin over the shared `ha0` floor) | ADE difference | verdict |
|---|---|---|
| RL-s0 vs BASE | RL worse by **0.083 m** [0.056, 0.115] | separated |
| NORL-s0 vs BASE | worse by **0.081 m** [0.051, 0.116] | separated |
| RL-s1 vs BASE | worse by **0.674 m** [0.555, 0.799] | separated |
| **RL-s0 vs NORL-s0** | **−0.0012 m** [−0.0285, +0.0268] | **not separated** |

⇒ **H-DDV2RL-2 = FAIL-HARM** (both RL seeds worse than the cold start). **The harm is the IL term's:** the control reproduces it without any policy gradient.
**Tactical family:** trajectory-implied lateral correctness BASE 0.950 → RL-s0 0.932 → NORL-s0 0.906; longitudinal 0.825 → 0.820 → 0.777. **Exploratory (no pre-registered criterion, no multiplicity correction):** RL-s0 beats NORL-s0 on three T1 secondaries — heading MAE by 0.33° [0.15, 0.54], tactical lateral by 2.6 pp [0.4, 5.4], tactical longitudinal by 4.3 pp [2.0, 7.0]. **Strategic family:** the harness marks it UNAVAILABLE on this corpus; the route-head readout is **identical (0.7917) in every arm**, as expected with a frozen trunk — a further known-value control.

### 7.4 Which variance each interval answers

* **Clip-cluster / episode-cluster bootstrap** (every CI above): "would another draw of held-out clips say this?" It sees neither training-seed nor inference-noise variance.
* **Training seed:** two RL seeds, reported as a contrast (n = 2, no CI) — and they disagree qualitatively (one stable, one diverged), which is itself the finding. **NORL has one seed**, so every RL−NORL difference carries NORL's unmeasured seed variance.
* **Inference noise:** paired per window in the T0 read, so it cancels in those differences; the T1 rolls all use infer-seed 0 (registry floor for this checkpoint ≈ 0.0001 m ADE).
* **I6 (known value) PASS:** the tip code reproduces the banked landing dump within **0.0014 m ADE [−0.0112, +0.0063]** on 308 shared windows over 9 episodes (the two rolls' stride phases coincide on 9 of 41 clips) — the ≤ 0.005 m bar holds.

## 8. What is NOT claimed

* **No PDMS number.** The reward is a proxy: no drivable area in training, EP normalised against the human instead of PDM-Closed, NC/TTC without their map clauses.
* **No closed-loop claim.** T1 is self-action OPEN loop (PI ruling 2026-09-02).
* **No claim about the paper.** This tests the recipe transplanted to a different state space, model, corpus, reward and scale.
* **No leaderboard entry.** The three checkpoints trained on 100 of the 141 EVAL clips and are burned.

## 9. Rule Zero — the next lever, and what it costs

**The measured failure mode is the release's all-modes IL pull**, not its policy gradient: it alone collapses the fan (37.5 → 2.45 m) and produces the whole T1 regression (+0.081 m of the +0.083 m). Intra-anchor GRPO needs within-group variation to rank; after the collapse there is almost none.

1. **LEVER 1 — dev box, ≈ 3 GPU-h, no pod.** Keep every release element except the IL form: replace the all-modes L1 with the **mode-preserving nearest-anchor** objective the REF-C trainer already uses (`refc_v3_train.py:2383-2391`, plus the `w_u0` control term), or λ ≈ 0.01, **and add gradient clipping 1.0** (the library's default; the release clips nothing and one of our two seeds diverged). Arms: RL vs NORL, two seeds each, same split, same endpoints. **Decision rule: only if Δfan separates from zero does anything downstream make sense.**
2. **LEVER 2 — on-support labels.** Train the sampler with DiffusionDrive's random-t objective (`sampler_train_t_max` is a dead flag) so labels 18…2 mean something, or run the RL chain on the deployed [10, 0] ladder as a declared deviation. D3 measured the off-support cost at ≈ 6.1 m of fan endpoint shift; the clamps add ≈ 5.7 m more (LEVER 2b: clamps off). A retrain is a pod job of refcv5-v2's own length (≈ 40 k steps).
3. **LEVER 3 — scale, only after LEVER 1 separates.** The release's 10 epochs at batch 512 over the ≈ 109 k joined B1-train windows. **ESTIMATED from the measured 0.621 s per window (0.296 s trunk+IO, 0.325 s chain+reward+grad):** on an A40-class GPU (assumed 2.5× this 4060) ≈ **75 GPU-h uncached**, or ≈ **39 GPU-h + 3.6 GPU-h** to build a frozen-trunk feature cache (≈ 7 GB). On this 4060 it would be 188 GPU-h.

## 10. UNVERIFIED

1. The **diffusers version the authors ran** — A18 and `final_alpha_cumprod = 1.0` are MEASURED on 0.40.0 here; the release pins nothing in the bank.
2. Everything tagged **PUBLISHED-CODE-UNBANKED** (SPEC §13): the NAVSIM-fork reward internals were read at the pinned commit through a model-transcribing fetch; the bytes are not hashed or banked. Re-banking them upgrades the class.
3. **Not read at all:** the Savitzky-Golay window/order actually used for comfort, `get_collision_type`'s exact thresholds, and how `_ego_areas` is filled (corners vs centre).
4. The **PhysicalAI ego box and pose origin** (nuPlan Pacifica numbers, rear-axle origin assumed). Controlled only indirectly by I4 (human NC = 1 on 100 % of windows).
5. The **mechanism of RL-s1's divergence** (trigger vs sustainer) — the ordering is measured, the causal chain is not.
6. **Why our fan collapse (−93 %) is so much larger than DDv2's reported diversity drop (−28 %, P Tab. 3)** — different diversity measures, horizons (6 s vs 4 s), anchor counts and scale.
7. **NORL's own training-seed variance** (one seed), and therefore the exact width of every RL−NORL difference.
8. Whether dev-box-scale conclusions transfer to release scale — untestable here by construction.
9. From the SPEC's own list: navtrain's sample count (hence the release's optimiser-step count), which eval script produced P Tab. 1's 91.2, and what P Tab. 3's "Top-K" ranks by.

## 11. Deliverable manifest

See the table printed by `code/verify_manifest.py` (staged blob vs worktree blob for every file) and §12 of the final report. The three arm checkpoints (1.3 GB each), the 5 held-out JSON reads' source copies, the T1 dumps and the harness logs live **only** under `C:/Users/Admin/tanitad-caches/ddv2rl-20260915/` on the dev box; the numeric extracts are banked in `raw/`. T1 JSONs and dumps are deliberately **not** banked: they carry clip ids.
