# RESULT — lever **L1 / D9**: mode-preserving imitation + gradient clipping on DDv2's RL stage

**Status on 2026-09-16: BUILT, TESTED AND PRE-REGISTERED. NOTHING HAS BEEN TRAINED.**
This file states **what will run and what will decide it**. It carries **no held-out number**, because none exists: the RTX 4060 belongs to another agent's proof package, and the four arms wait for the Master Mind's GPU slot. Every measured number quoted below is from the **2026-09-15** validation and is tagged as such.

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-16-refcv6-rl-l1/`
**Worktree:** `C:/Users/Admin/tanitad-wt-rl-l1` at `9a782fa` (tip of `agent/arch-inf-20260803`)
**PI instruction:** *"Do what is necessary to prepare the RL post training, implement D9"* (2026-09-16), executing `…/2026-09-15-ddv2-rl-prep/RESULT.md` §9 **LEVER 1**.
**Pre-registration:** `Project Steering/PREREG_DDV2_RL_VALIDATION.md` **§§11-16** (appended — 138 insertions, **0 deletions**; §§1-10 are byte-unchanged).
**Element-by-element proof that only two things changed:** `DIFF_L1_VS_RELEASE.md` in this package.

---

## 0. The scope limits that carry forward — none of them is relieved by this lever

1. **The three 2026-09-15 checkpoints are BURNED** — they trained on **100 of the 141 EVAL clips** — and **must never be scored on the 141-clip panel** or entered on a leaderboard. ⛔ **The four checkpoints this lever will produce are burned on exactly the same split and inherit exactly the same prohibition.** They are validation artifacts.
2. **The dev-box scale is ≈ 0.4 % of the paper's optimiser samples.** 600 steps × 4 windows = 2,400 windows ≈ 0.97 passes over the 2,464 eligible RL-train windows, against the release's 10 epochs of navtrain at batch 512. This is a mechanism validation, not a reproduction of scale — and L1 does not change that.
3. **The reward is a PROXY with NO DRIVABLE-AREA TERM** — our SAM3 maps do not yet cover the training clips, so DAC ≡ 1 and the release's DAC constraint branch is inert. NC/TTC lack their map clauses and EP is normalised against the human rather than PDM-Closed. ⛔ **Never quote any number from this package as PDMS.**
4. **T1 is self-action OPEN loop** (PI ruling 2026-09-02). Nothing here is a closed-loop claim.
5. **No claim about the paper.** This tests the recipe transplanted to a different state space (control, not metric waypoints), model, corpus, reward and scale.

---

## 1. What is being fixed, and on what evidence

**MEASURED 2026-09-15** (`…/2026-09-15-ddv2-rl-prep/RESULT.md`, `raw/heldout_*.json`, n = 493 held-out windows over 40 clips, deployed sampler, paired inference noise, clip-cluster bootstrap):

| | BASE (cold start) | the release's RL | the release's IL alone (NORL) | the diverged seed |
|---|---|---|---|---|
| fan endpoint spread (m) | **37.5199** [32.8, 42.0] | 2.4638 [2.06, 2.90] | **2.4457** [1.95, 2.99] | 19.6838 [17.6, 21.9] |
| retained fraction of the cold start | 100 % | 6.57 % | **6.52 %** | 52.46 % |
| fan best-of-117 ADE (m) | 0.888 | 1.532 | 1.543 | 3.664 |
| T1 ADE vs BASE (m, paired) | — | +0.083 [0.056, 0.115] | **+0.081 [0.051, 0.116]** | +0.674 [0.555, 0.799] |
| max grad norm over the run | — | 147.3 | 146.0 | **15,712.3** |
| median grad norm | — | 16.67 | 17.93 | 61.96 |
| steps a **1.0** clip would have bound on | — | **600/600** | **600/600** | **600/600** |
| steps a **100** clip would have bound on (the amended value) | — | **1/600** | **1/600** | **276/600** |

⇒ **The harm is the release's imitation term, not its RL term.** Removing only the policy gradient reproduces it (RL − NORL on T1 ADE: **−0.0012 m [−0.0285, +0.0268]**, not separated). And intra-anchor GRPO needs within-group variation to rank; after a 93 % collapse there is almost none, which is why the RL term measured as doing nothing.

**The lever, therefore, is exactly two changes** (`DIFF_L1_VS_RELEASE.md` §1):

* **C1** — the all-modes L1 over all 468 chains is replaced by the **mode-preserving matched-anchor** L1 our own trainer already uses (`stack/scripts/refc_v3_train.py:2383-2391`): the imitation signal lands on the G chains of the anchor nearest the GT, and the other 464 receive none.
* **C1b** — the **plain λ ≈ 0.01** variant of the release's own form is exposed beside it as the cheaper alternative arm, so the two are compared rather than chosen by argument. It doubles as C1's **confound control**: it answers whether a surviving fan is due to mode preservation or merely to less imitation pressure.
* **C2** — **gradient clipping, max-norm 100**, once, before `opt.step()`. The release clips nothing. *(Was 1.0; see AMENDMENT A-1.)*

### ⭐ AMENDMENT A-1 — the clip max-norm, **1.0 → 100**, made BEFORE any arm ran

**The escalation.** C2 was first implemented at the instructed **1.0**. **MEASURED on the banked logs** (`…/2026-09-15-ddv2-rl-prep/raw/metrics_arm-*.jsonl`, 600 steps × 3 arms): a max-norm of 1.0 **would have bound on 600/600 steps of all three arms** — minimum norm ever observed **1.87 / 2.22 / 3.45**, medians **16.67 / 17.93 / 61.96**. So 1.0 does not merely catch RL-s1's spike: it is a **17-62× rescale on every single step**.

**The ruling.** The coordinator **verified these numbers independently against the same three files** and upheld the escalation on 2026-09-16: `> 1.0` binds 600/600 on all three arms; `> 100` binds **1/600** on each stable arm and **276/600** on the diverged seed (max 15,712.3). *"1.0 was my number and it was wrong. You were right to implement it as instructed and escalate rather than silently substitute."* **The pre-registered max-norm is now 100** — an actual spike guard, which is what C2 is for.

**Recorded as a PRE-RUN amendment.** At amendment time **no arm had run**: zero GPU-minutes on this lever, no L1 checkpoint, no L1 held-out number. `PREREG_DDV2_RL_VALIDATION.md` §11 **AMENDMENT A-1** carries the reason, the verification table, the date and that fact; `ddv2_il.GRAD_CLIP` and `--grad-clip`'s default are now `100.0`, pinned by `test_AMENDMENT_A1_the_default_clip_is_a_spike_guard_not_an_every_step_rescale`.

**Scope of A-1: a single scalar.** Nothing else moved — the **F1 gate is not re-derived** (still CI lower bound > −15.01 m, ≥ 60 % of 37.5199 m), nor `Δfan`, the arms, split, cold start, seeds, steps, batch, I1-I10, budget or scope limits. `--grad-clip 1.0` and `--grad-clip 0` (the release) remain available as declared arms; **1.0 becomes interesting only if the diverging seed also fails at 100**, and running it would be a new arm with its own record.

**What is still only REASONED.** Under AdamW a *constant* rescale of the whole gradient largely cancels in `m/(√v + ε)` — an argument about the optimiser that this package does not test. It is why 1.0's every-step behaviour could have passed unnoticed. The rescale is not constant across steps and Adam's `v` is an EMA across them, so a clip does change the trajectory; §4.4 reports `grad_norm`, `grad_clipped` and `param_delta_norm` against the banked arms so the change is measured rather than assumed, and **I10** still refuses a dead clip flag at 100.

Everything else is the release, unchanged **by construction**: `git diff` on `stack/tanitad/rl/ddv2_rl.py`, `ddv2_refc_chain.py` and `pdm_proxy.py` is **empty**; the only edited tracked file that CHANGES BEHAVIOUR is the driver (77 insertions / 11 deletions); `stack/tests/test_ddv2_rl.py` is also edited, for the banked-byte-pin fix (§8.5), which touches no product code.

---

## 2. What has already been established, on CPU, before any GPU time

| | evidence |
|---|---|
| the 2026-09-15 machinery still passes | **74/74** on this worktree, with NO environment override (`test_ddv2_rl.py` 39 — 38 pre-existing + the CRLF mutation proof — `test_ddv2_refc_chain.py` 10, `test_pdm_proxy.py` 25) |
| the lever's own suite | **42/42** (`stack/tests/test_ddv2_il.py`) — **116 passed** together |
| ⭐ **the mode-preserving term really preserves modes** | `test_MUTATION_a_synthetic_fan_survives_the_matched_term_and_does_NOT_survive_the_release`: a synthetic 9-anchor fan of initial endpoint spread **23.70 m**, optimised 400 Adam steps under each term alone. The **release's** term → **0.060 m (0.25 % retained)**; the **matched** term → **23.7037 m (100.0 % retained)**; separation **393×**. All three are *required*, so the defect is proven **reachable** before the guard is credited |
| and by mechanism, not only by statistic | `test_MUTATION_only_the_matched_anchors_chains_moved_at_all`: every **unmatched** anchor's chains are **bit-unchanged** (max displacement exactly `0.0`) under the matched term, and **all** of them move under the release's |
| clipping is the measured magnitude | `test_clip_reports_the_PRE_clip_norm_and_binds_at_the_measured_divergence`, parametrised over **100** (the amended default) and **1.0** (still an available arm): a gradient of norm **15,712** (RL-s1's observed max) becomes exactly the max-norm with its direction preserved; the pre-clip norm stays the logged one |
| **A-1 cannot drift back silently** | `test_AMENDMENT_A1_the_default_clip_is_a_spike_guard_not_an_every_step_rescale` pins `GRAD_CLIP == 100.0`, and shows the banked **median** step (norm 16.67) is now **untouched** where 1.0 would have rescaled it |
| the banked-source byte pin survives any checkout | `test_MUTATION_the_normalised_bank_hash_still_catches_a_one_character_change` (§8.5): a CRLF checkout of the correct content **passes**; a one-character change, a whitespace-only change and a one-byte deletion each go **red** |
| clipping is in the right place | `test_REGRESSION_clipping_per_rollout_step_is_not_clipping_the_update`: clipping each of the 10 partial gradients leaves a total norm > 1.5; clipping once before `opt.step()` gives exactly 1.0 |
| the run cannot lie about which recipe ran | `check_arm_l1.py` I7-I10, each proven to fire on its own defect — including **an arm that declares `matched` and ran the release** |

⛔ **What none of this establishes:** that the fan survives on the REAL model, that RL then separates, or that the T1 harm goes away. Those are the GPU arms' job and are pre-registered below.

---

## 3. The arms that will run

All arms: `--steps 600 --batch 4`, release defaults otherwise, the **same split** as 2026-09-15 (`raw/SPLIT_eval141_sha12.json`: 41 held-out clips / 100 RL-train clips, fixed by `sha256(clip_id)` order), the same cold start (`C:/Users/Admin/refcv5v2_final/ckpt.pt`, md5 `9405ec73b2d797c4cebd44f82dbce54b`, step 40,284; the driver refuses a different one).

### Stage A — the primary, four arms

| arm | command | differs from **L1-RL-s0** by |
|---|---|---|
| **BASE** | the cold start, untrained | no training |
| **L1-RL-s0** | `train --arm rl --seed 0 --il-form matched --grad-clip 100` | — |
| **L1-NORL-s0** (length-matched control) | `train --arm norl --seed 0 --il-form matched --grad-clip 100` | the policy-gradient coefficient is **exactly 0** on every step; the IL weights stay the release's advantage-derived 0.1 / 1.0 |
| **L1-RL-s1** (replicate) | `train --arm rl --seed 1 --il-form matched --grad-clip 100` | window order and chain noise |

### Stage B — the cheaper alternative arm, CONDITIONAL on budget

| arm | command | what it answers |
|---|---|---|
| **L1λ-NORL-s0** | `train --arm norl --seed 0 --il-form lambda --grad-clip 100` | the release's **form** at **λ ≈ 0.01**, IL only. ⭐ **Was the fan saved by mode preservation, or merely by less imitation pressure?** The IL-only arm is the shortest path to that question because 2026-09-15 MEASURED that the collapse and the whole T1 regression are the IL term's alone |

**Run condition (pre-registered):** Stage B runs **only if** the cumulative GPU time after Stage A is ≤ **2.45 h**. Otherwise it is recorded as **NOT RUN**, never as "unnecessary".

### Held-out reads

`ddv2_rl_refcv5.py heldout --stride 10` on the 41 held-out clips, **deployed** sampler (2 steps, η = 0, one sample per anchor), **identical inference noise per window across every checkpoint** (`seed = 500000 + k`) — for BASE, BASE-repeat (control **I5**), and every trained arm. Then `refcv3_arm.py` T1 rolls (grid 2 s, stride 5, infer-seed 0) for BASE and the three Stage-A arms, with all **four metric families**.

---

## 4. What decides it

### 4.1 The gate — **F1, the fan must not collapse**

Primary quantity: `Δspread(X) = fan_endpoint_spread_m(X) − fan_endpoint_spread_m(BASE)`, **paired per window** with identical inference noise, **clip-cluster bootstrap** (2,000 resamples, seed 0, 95 % percentile CI), on the same 493-window / 40-clip held-out read.

> **F1 PASS** for arm X iff the 95 % CI **lower** bound of `Δspread(X)` is **> −15.01 m**, i.e. X retains **≥ 60 %** of the cold start's **37.5199 m**.
> **F1 FAIL** iff the CI **upper** bound is below that line. Otherwise **F1 INCONCLUSIVE**.

**Why 60 %, stated against the two reference points the PI named:**

| reference | spread (m) | retained | vs the bar |
|---|---|---|---|
| cold start (BASE), MEASURED | **37.5199** | 100 % | — |
| **the measured failure** (the release's IL alone), MEASURED | **2.4457** | **6.52 %** | fails by **53 pp** — the bar cannot be passed by the thing it exists to refuse |
| the diverged seed RL-s1, MEASURED | 19.6838 | 52.46 % | **also fails** — a bar this programme's own known-harmful arm could squeak past would be worthless |
| DDv2's own reported diversity drop (P Tab. 3, −28 %) | — | ≈ 72 % | **passes** — the bar does not forbid the sharpening the paper claims is desirable |
| **the bar** | **22.51** | **60 %** | |

**F1 is a gate, not an endpoint.** It is evaluated for L1-RL-s0, L1-NORL-s0 and L1-RL-s1, and **it is read before the primary endpoint**: if the fan collapses again, the lever is refuted and `Δfan` is reported but decides nothing.

### 4.2 The primary endpoint — **Δfan must separate**

`Δfan(X) = fan_pdms_mean(L1-RL-sX) − fan_pdms_mean(L1-NORL-s0)`, X ∈ {0, 1} — the **same** endpoint, pairing and statistic as the 2026-09-15 prereg §5.2.

> **SUCCESS** — L1-RL-s0, L1-NORL-s0 and L1-RL-s1 all pass **F1**, AND `Δfan(0)` and `Δfan(1)` both have 95 % CI lower bounds **> 0**.
> **PARTIAL** — F1 passes for all three, exactly one seed's `Δfan` separates, the other's CI contains 0.
> **FAILURE** — F1 fails for any trained arm, OR both `Δfan` CIs contain 0, OR either upper bound < 0.

⛔ **The decision rule from LEVER 1, restated:** *only if `Δfan` separates from zero does anything downstream make sense.* A FAILURE here means the RL term still adds nothing on this model at this scale, and the next lever is **LEVER 2 (on-support labels)**, not more of this one.

⚠️ **A cross-package comparison that is PRE-REGISTERED AS INVALID.** `fan_pdms_mean` is a mean over 117 members, so **collapse raises it** by making the members alike (MEASURED: NORL−BASE **+0.2205 [0.1836, 0.2562]** *while* the fan's best member and its coverage of the human got **worse**). ⛔ An L1 arm's `fan_pdms_mean` must therefore **never** be compared with a 2026-09-15 arm's as if higher were better. It is compared only within this package, against a control with the same spread, and it is always reported beside the spread, the oracle and the best-of-117 ADE.

### 4.3 The harm guard — **H-DDV2RL-2**, unchanged

> **FAIL-HARM** if, for **both** RL seeds, the T1 `ade_m` of `os` vs BASE has a paired (episode-cluster bootstrap) CI lower bound **> 0** (worse than the cold start). Otherwise "no harm detected at this n" — which is not a claim of no harm.

### 4.4 Reported with a direction but **no criterion**

The 2026-09-15 collapse hurt exactly these, so they are the mechanism's fingerprints: fan **best-of-117 (oracle)** proxy (MEASURED 0.994 → 0.956) and **best-of-117 ADE** (0.888 → 1.532 m) are expected to stay near BASE if the fan survives; `sel_*` deltas; `NORL − BASE` (the IL term's own effect under the new form); the fan collision fraction; the clamp-binding rates; and the within-run canaries `chain_endpoint_spread_m`, `il_all_modes_m` vs `il_matched_anchor_m`, `match_n_distinct_anchors_matched`, `grad_clipped`.

**Mechanism evidence for "RL now has something to rank"** (reported, no criterion): `frac_positive_after_bar` and the within-group reward spread over training, against the 2026-09-15 arms' logged values (RL-s0 4.2 % → 6.4 %).

**The clip's own effect** (reported, no criterion, per the escalation in §1): the fraction of steps on which the clip bound (**expected ≈ 100 %** from the banked logs), the pre-clip `grad_norm` distribution, and `param_delta_norm` against the banked arms' (RL-s0 **7.81**, NORL-s0 **8.56**, RL-s1 **11.03** at step 600). A `param_delta_norm` far below those would say the every-step clip changed the optimiser's trajectory materially — which the AdamW argument predicts it will not, and which this diagnostic is here to check rather than assume.

### 4.5 Integrity — all must hold, else the run is **VOID**, not negative

**Carried forward unchanged:** **I1** every step finite, parameters moved · **I2** NORL's policy-gradient coefficient exactly 0.0 with the logged `rl_part` inside float32 round-off (the D-1 amended form) · **I3** RL arms carry a positive advantage on ≥ 90 % of steps · **I4** the human's own proxy NC = 1 on ≥ 98 % of training windows · **I5** two BASE held-out reads with the same seeds agree **bitwise** · **I6** BASE T1 vs the banked landing dump within 0.005 m ADE, or reported **NOT RUN**.

**New, and each one proven able to fail by mutation** (`check_arm_l1.py`, `test_MUTATION_each_L1_integrity_check_fires_on_its_own_defect`):

* **I7 — the lever is on, and is the one claimed.** `run.json`'s `il.form` / `il.lambda_scale` / `grad_clip` match the flags, and **every** logged step agrees with them.
* **I8 — the matched term is the one that drove.** On a matched arm `il_mean_m == il_matched_anchor_m ≠ il_all_modes_m`; on a release/λ arm `il_mean_m == il_all_modes_m`.
* **I9 — the match is not degenerate.** Not every batch matched a single anchor on every step.
* **I10 — clipping is real and is not everything.** No step's post-clip norm exceeds 1.0; and either the clip bound on some step, or the run's peak pre-clip norm stayed under 1.0 (so "never clipped" is measured, not a dead flag).

---

## 5. Budget — ≤ 3.0 GPU-h on the RTX 4060

**MEASURED** from the banked logs' own wall clock: **2.60 / 2.49 / 2.46 s per step** (RL-s0 / NORL-s0 / RL-s1). *(§8 of the 2026-09-15 prereg budgeted 2.7 s — that was the smoke estimate; the completed runs came in below it.)* L1 adds, per step, two `no_grad` L1 evaluations on an existing tensor, one 117×117 `cdist` and one `argmin` over `[B, 117, 8]` ⇒ **ESTIMATED ≈ 2.65 s**. The table below keeps the conservative **2.75 s**, so the totals are an upper bound.

| item | n | each | total |
|---|---|---|---|
| Stage A training (L1-RL-s0, L1-NORL-s0, L1-RL-s1) | 3 | ≈ 27.5 min | **82.5 min** |
| held-out T0 reads (BASE, BASE-repeat, 3 arms) | 5 | ≈ 4 min | **20 min** |
| T1 rolls (BASE + the 3 arms) | 4 | ≈ 11 min | **44 min** |
| **Stage A subtotal** | | | **146.5 min = 2.44 GPU-h** |
| Stage B (L1λ-NORL-s0 + its T0 read), CONDITIONAL | 1 | ≈ 31.5 min | **0.53 GPU-h** |
| **Total** | | | **≈ 2.97 GPU-h** |

**Drop order, pre-registered**, if the cumulative passes **2.85 h**: (1) Stage B; (2) L1-NORL-s0's T1 roll; (3) L1-RL-s1's T1 roll — and each drop is recorded as **NOT RUN**, with the criterion it disables named. Dropping (3) makes **H-DDV2RL-2 unevaluable** and the guard is then reported as such, never as passed.

**Already spent on this lever: 0 GPU-minutes.** Implementation, all 110 tests and this pre-registration ran on CPU.

---

## 6. How to run it, once the slot is granted

```sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WT=/c/Users/Admin/tanitad-wt-rl-l1
OUT=/c/Users/Admin/tanitad-caches/ddv2rl-l1-20260916
PKG="$WT/TanitAD Research Lab/Architecture & Inference/Research/2026-09-16-refcv6-rl-l1"

# Stage A — one arm; repeat for (norl,0) and (rl,1)
"$PY" stack/scripts/ddv2_rl_refcv5.py train --arm rl --seed 0 --steps 600 --batch 4 \
      --il-form matched --grad-clip 100 --out-dir "$OUT/l1-rl-s0" > "$OUT/l1-rl-s0.log" 2>&1
"$PY" "$PKG/code/check_arm_l1.py" "$OUT/l1-rl-s0" --steps 600 --arm rl \
      --il-form matched --grad-clip 100        # exit 0 = I1/I3/I7-I10 all hold

# the held-out T0 read (BASE twice, for I5)
"$PY" stack/scripts/ddv2_rl_refcv5.py heldout --ckpt "$OUT/l1-rl-s0/ckpt.pt" \
      --config "$OUT/l1-rl-s0/config.json" --stride 10 --out "$OUT/heldout_l1-rl-s0.json"
```

⛔ **Never chain the lander behind a pipe** and **never `git add <dir>` on `agent/arch-inf-*`** (MEMORY). ⛔ Check the GPU is free (`nvidia-smi --query-compute-apps`) before every arm: another agent owns the card for a proof package.

---

## 7. Deliverable manifest

| file | state | in one place only? |
|---|---|---|
| `stack/tanitad/rl/ddv2_il.py` | **NEW** — the two changes, and nothing else | ✅ the only home of the lever's arithmetic |
| `stack/tests/test_ddv2_il.py` | **NEW** — 42 tests incl. the fan mutation proof and A-1's pin | ✅ |
| `stack/scripts/ddv2_rl_refcv5.py` | **EDITED** — +77 / −11, the driver only | ✅ |
| `stack/tests/test_ddv2_rl.py` | **EDITED** — +73 / −5: the banked byte pin now hashes line-ending-normalised bytes (same pin value) + its mutation proof | ✅ |
| `Project Steering/PREREG_DDV2_RL_VALIDATION.md` | **EXTENDED** — §§11-16 appended (138 insertions, 0 deletions), §§1-10 byte-unchanged | ✅ the pre-registration, extended not rewritten |
| `…/2026-09-16-refcv6-rl-l1/DIFF_L1_VS_RELEASE.md` | **NEW** — element-by-element vs the release | ✅ |
| `…/2026-09-16-refcv6-rl-l1/RESULT.md` | **NEW** — this file | ✅ |
| `…/2026-09-16-refcv6-rl-l1/code/check_arm_l1.py` | **NEW** — I1/I3/I7-I10, each proven to fire | ✅ |
| `stack/tanitad/rl/ddv2_rl.py`, `ddv2_refc_chain.py`, `pdm_proxy.py` | **UNCHANGED** (`git diff` empty) | — |

**Staged, never committed, never pushed**; branch `agent/arch-inf-20260803` is not switched and nothing outside the eight paths above is staged.

---

## 8. UNVERIFIED / open

1. **Everything about the outcome.** No L1 arm has been trained; §4's verdicts are empty.
2. **Whether the matched form's concentrated budget is the right scale on the real model.** The normalisation choice is argued and tested (`DIFF` §1), but 117× fewer supervised chains at the same total budget is a *different* per-chain pressure, and only the run measures whether it is too much. The canary is `il_matched_anchor_m` collapsing toward 0 while `chain_endpoint_spread_m` falls — reported every step.
3. **Whether a preserved fan actually gives GRPO within-group variation to rank.** This is the lever's whole premise and it is a hypothesis, not a measurement: the fan spread is *across* anchors, while the advantage is normalised *within* each anchor's G = 4 samples. §4.4's mechanism evidence is what will test it.
4. All nine UNVERIFIED items of `…/2026-09-15-ddv2-rl-prep/RESULT.md` §10 carry forward unchanged — in particular the diffusers version the authors ran, the unbanked NAVSIM-fork reward internals, the PhysicalAI ego box, and why our collapse (−93 %) so far exceeded DDv2's reported −28 %.
5. ~~An environment finding left unfixed~~ — **FIXED 2026-09-16 on the coordinator's instruction, and the fix is proven by mutation.** The banked-release byte pin (`test_ddv2_rl.py::test_the_excerpt_matches_the_banked_release_when_the_bank_is_checked_out`) hashed **raw** bytes, so with the dev box's `core.autocrlf=true` a fresh **non-sparse** worktree read `a9075847ee…` instead of the pinned `2789081af3b6…` — the content was right (`git cat-file -p HEAD:<path> | sha256sum` = the pin exactly, verified by both of us) and only the checkout differed. The test previously passed only because predecessor worktrees were **sparse** and took its skip path, so a full checkout would have gone red on the platform and taught the reader to re-baseline on red. It now hashes **line-ending-normalised** bytes (`raw.replace(b"\r\n", b"\n")`), keeping the **same pin value** — the repo's existing convention (`tests/test_v5_trainer_v2_val.py:880`). ⛔ **Proven still a real guard, by mutation:** a CRLF expansion of the correct content **passes**, and a one-character change (`class` → `Class`, same length), a whitespace-only change and a one-byte deletion each go **red** — asserted in `test_MUTATION_the_normalised_bank_hash_still_catches_a_one_character_change`, and demonstrated end-to-end by pointing `TANITAD_BANK_ROOT` at a one-character mutant, which failed the guard with `a86fe78ff1d6…`. No `.gitattributes` change was needed, so no live worktree is disturbed.
