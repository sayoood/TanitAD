# E-WC2 — σ\* MEASURED. **SEL-1 is REFUSED**, at 3.3× past the pre-registered refusal line

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Cost: 0 GPU. Thor untouched.**
**Pre-registration:** `Project Steering/V6F_PLANNER_DESIGN.md` §5.2 (outcomes committed in advance) + §5.3
**Instrument:** `stack/scripts/e_wc2_sigma_star.py` (built + 62 tests in `5daa3d7`; run here for the first time on real data)

---

## 0. The one-line answer

> **σ(2 s) = 4.7104 m per-axis on the REF-C-XL surface → σ/ADE = 9.9915.**
> §5.2 committed **σ/ADE ≥ 3.0 ⇒ SEL-1 is refused before launch.** The verdict is
> **REFUSED**, and it is not close: the interval's **lower** bound, 7.4492, is still
> **2.5× past** the refusal threshold. Both REF-C arms agree.
> §5.3's refutation check **also fires**: σ(6 s) = 3.75 × σ(2 s) > 3 ⇒ **REDERIVE** —
> no scaled 6 s threshold is emitted, by either arm, under any branch.

⚠️ **AND THE MORE ACTIONABLE HALF, which the verdict alone hides.** A **0-parameter
constant-yaw-rate extrapolation** reaches **σ(2 s) = 1.1888 m** on the same 881 windows —
**3.96× better than the ridge on frozen REF-C vision latents.** So the finding is *not*
"a 6 s goal is unpredictable on this corpus". It is **"the frozen REF-C latents are the
wrong surface for it"**. That distinction changes what the fallback should be — see §6.

---

## 1. What was missing, and why this was reachable for 0 GPU

The instrument was complete. Exactly one input was absent: **`gt_endpoint` / `endpoint_steps`
— the 6 s ground-truth endpoint** (`refc_dump_latents.py` grew `--backfill-endpoints` for
precisely this). The endpoint is GROUND TRUTH FROM EGO POSES: no model, no GPU. The only
real prerequisite was **the val40 pose arrays on a reachable disk**, and the standing belief
was that they lived only on Thor — which is training a 336M model and was off-limits.

**They are also on HuggingFace**, in the mirror of the 256 px val epcache:

| | |
|---|---|
| repo | `hf:datasets/Sayood/tanitad-physicalai-w120-256x640cyl` |
| path | `epcache-256px-phase0/physicalai-val-0c5f7dac3b11/ep_{00000..00039}.pt` |
| size | 117,383,256 B × 40 = **4.70 GB** |

⛔ **I did not move 4.70 GB.** A `torch.save` file is an **uncompressed zip**, so `poses`
is a ~3 KB member inside a 117 MB archive. `code/hf_poses_pull.py` reads the central
directory and `data.pkl` over **HTTP Range**, identifies the `poses` storage from the
pickle without materialising a tensor, and fetches only that member.

| | MEASURED |
|---|---|
| bytes fetched | **18,419,392 B** |
| bytes if whole files | 4,697,689,792 B |
| **saving** | **255.0×** |
| wall clock | **81.1 s**, 40/40 episodes |

**Parity is proved, not asserted.** Every episode's extracted poses bytes are sha256'd
against the **committed** `manifest_EVALPOD_val40.json`, which recorded
`sha256(poses.numpy().tobytes())` per episode off the canonical cache:

| check | result |
|---|---|
| `poses_sha256` vs the committed manifest | **40 / 40 bit-identical, 0 mismatches** |
| `episode_id` and `T` vs the same manifest | **40 / 40 identical** |
| windows from those `T` on the producer's grid | **881** — the canonical count |

*(Artifact: `raw/val40_hf_poses_verify.json`. Precedent: `…/2026-08-04-distance-keeping-arms/`
did the same reconstruction from a Thor relay and hit the same 40/40 on 2026-08-04.)*

⚠️ **Evidence class: MEASURED (ours)** — `raw/val40_hf_poses_verify.json`.

---

## 2. ⛔ The alignment gate fired — and it was RIGHT to fire and WRONG to refuse

The backfill refuses unless the recomputed 2 s endpoint matches the banked fan's `gt`
column, because a one-window slip regresses every latent onto a **neighbour's** endpoint
and returns an inflated σ *that looks exactly like a measurement*. It refused:

```
recomputed endpoint at step 20 is not bit-identical to the banked gt column
```

**MEASURED, before changing anything** (`raw/endpoint_backfill_controls_refc-*.json`):

| | value |
|---|---|
| rows bit-identical | **825 / 881** |
| max aligned error | **7.63e-06 m** |
| …in ULPs of the row magnitude | **1.118** |
| ±1-row shifted error (median) | **0.5123 m** ≈ **4e5 ULPs** |
| **separation** | **67,139×** |

The dumps were produced **on Thor (aarch64)**; the backfill runs on the **x86 dev box**.
`gt_ego_waypoints` rotates by `cos(-yaw)/sin(-yaw)` and the two libms disagree in the
**last bit**. A literal `torch.equal` therefore refuses a **correct** backfill — and the
"fix" that suggests itself (re-run the dump on x86) costs a GPU pass for a rounding mode.

⭐ **I did not weaken the gate; I made it strictly stronger.** It now requires **both**:

1. agreement within `ENDPOINT_ULP_TOL = 4` ULPs **of the row's magnitude** — and the row
   normalisation is itself load-bearing: a rotation spreads the *longitudinal* magnitude's
   last bit into the near-zero *lateral* component, where a per-**component** ULP count
   reads this same 1-ULP error as **256 ULPs** and refuses correct data;
2. ⭐ **a ±1-row POSITIVE CONTROL** — the shifted alignments must be ≥ **1000×** worse.
   **This did not exist before.** A bare bit-identity check passes *vacuously* on a
   degenerate block (a parked ego, an all-zero stand-in) where every shift also matches,
   and carries no evidence about alignment at all. The new gate refuses that case
   (`test_backfill_gate_is_not_vacuous_on_a_degenerate_block`).

Five tests pin it, including `test_backfill_still_refuses_a_one_row_shift` — the failure
the tolerance must never admit. `stack/scripts/refc_dump_latents.py`, `stack/tests/test_e_wc2_sigma_star.py`.

*(Root-cause class, for `RETRACTION_LOG.md`: **an exactness check whose exactness is a
property of the MACHINE, not of the claim.** Bit-identity is the right assertion within one
process and the wrong one across two ISAs; the durable form is "agreement at the last bit
**plus** a control that the wrong answer would fail". Sibling of C14 — the gate reported the
shape of its own arithmetic and would have been read as the shape of the data.)*

---

## 3. The pre-registered surface — met, and checked by measurement

| §5.2 requirement | delivered | MEASURED where |
|---|---|---|
| 40 val episodes | **40** | `raw/ewc2_sigma_star_refc-*.json` `surface` |
| the canonical grid | **881 windows** | same; `controls_vs_bank.n_windows_match: true` |
| the banked fan is the same fan | `fan_bit_identical: true`, `fan_max_abs_diff: 0.0` | same |
| producer clean | `instrument_fail: []`, `contract_problems: []` | same |
| **LOEO = leave-one-EPISODE-out** | **40 folds, 1 episode/fold, 0 straddlers, 22–23 windows/fold** | `raw/ewc2_preconditions_verified.json` |
| σ at 2 s **and** 6 s | both | §4 |
| features **VISION_ONLY** | `pooled` (992) + `ctx` (96) = 1088 dim; `any_echo: false` | `features` block |

### 3a. ⚠️ LOEO is episode-disjoint — verified on the REAL dump, not on a fixture

A window-disjoint split puts a window's stride-8 **neighbours** in train (the REF-A I-JEPA
defect) and biases σ **down** — on precisely the number this experiment exists to produce.
Re-running the actual estimate under both schemes on the actual latents:

| horizon | σ LOEO | σ window-disjoint | understatement |
|---|---|---|---|
| 2 s (XL) | **4.7104** | 2.2903 | **2.06×** |
| 6 s (XL) | **18.3519** | 9.5133 | **1.93×** |
| 2 s (base) | **4.5545** | 2.2301 | **2.04×** |
| 6 s (base) | **16.1473** | 8.9511 | **1.80×** |

⇒ A leaked E-WC2 would have reported σ/ADE ≈ 4.86 instead of 9.99. **Still REFUSED** — but
the leak is real, is ~2× on this dump, and is not taken.

### 3b. ⚠️ σ IS PER-AXIS, NOT RADIAL — pinned by reproducing §3.1's own published ratios

§3.1 injects `g = gt_end + rng.normal(0.0, s, size=gt_end.shape)`
(`sel_winners_curse_law.py:221`), so `s` is the **per-axis** SD of an isotropic 2-D
Gaussian. Reading the **radial RMS** against the same threshold inflates σ by **√2 = 1.4142**.

| reading | σ\*/ADE | σ\*/oracle | reproduces §3.1's published **1.7 / 4.9**? |
|---|---|---|---|
| **per-axis** (0.8 m) | **1.6971 → 1.7** | **4.8810 → 4.9** | ✅ **yes** |
| radial RMS (1.1314 m) | 2.4000 | 6.9028 | ❌ no |

⇒ The unit is pinned by the published numbers themselves. Both forms travel in every
result JSON so they cannot be silently swapped. *(`raw/ewc2_preconditions_verified.json`.)*

---

## 4. THE NUMBERS

**Estimator:** point estimates are **full-set**; intervals are the **episode-cluster
bootstrap** over the 40 val episodes (`taniteval/ci.py`), 2000 draws; the ratio interval
uses **ONE shared episode draw** for numerator and denominator. ⛔ `overlapping_holdout_se`
is used nowhere.

### 4a. σ, per-axis, metres

| arm | horizon | **σ per-axis** | 95 % CI | σ radial RMS | σ long | σ lat | n windows | n episodes |
|---|---|---|---|---|---|---|---|---|
| **refc-xl-30k** | **2 s** | **4.7104** | **[3.8087, 5.6860]** | 6.6615 | 6.5752 | 1.0688 | **881** | 40 |
| **refc-xl-30k** | **6 s** | **18.3519** | **[15.8621, 20.9608]** | 25.9534 | 23.2124 | 11.6089 | **681** | 40 |
| refc-base-30k | 2 s | 4.5545 | [3.5487, 5.6072] | 6.4411 | 6.3723 | 0.9391 | 881 | 40 |
| refc-base-30k | 6 s | 16.1473 | [13.7968, 18.5165] | 22.8357 | 19.9523 | 11.1075 | 681 | 40 |

**The 6 s n is 681, not 881, and that is a reported exclusion, never an imputation.**
200 windows lie within 6 s of their episode's end (`endpoint_valid` = 0.773); they are
masked NaN and dropped per horizon with the reason and the n. The window grid was **not**
widened to reach 6 s — that would have re-selected windows and broken both the 881-window
parity and the fan bit-identity gate.

### 4b. The ratios — the transferable claim

| arm | **σ/ADE** | 95 % CI | **σ/oracle** | incumbent ADE | oracle ADE | §3.1 reference |
|---|---|---|---|---|---|---|
| **refc-xl-30k** | **9.9915** | **[7.4492, 13.5119]** | **28.7307** | 0.4714 | 0.1639 | σ\*=0.8 ⇒ 1.7 / 4.9 |
| refc-base-30k | 9.6337 | [7.0191, 13.2303] | 23.7933 | 0.4728 | 0.1914 | " |

σ(2 s) is **5.89× σ\***. **σ/ADE is 3.33× the refusal threshold, and the CI's lower bound is
2.48× it.** There is no reading of this interval that reaches even the INCONCLUSIVE band.

⭐ **The denominators are §3.1's own, not re-derived.** The XL run's reference block reproduces
§3.1's published fan references **exactly** — oracle **0.1639**, fan mean **13.9564**, shipped
supervised selector **0.4714** — because it reads the same banked fan
(`…/2026-08-03-esel-verdict/raw/fan_refined_refc-xl-30k.pt`, 881 windows / 40 episodes).
So σ/ADE is a ratio of a newly measured numerator to a **published, unaltered** denominator.

### 4c. §5.3 refutation check — **REDERIVE**, both arms

| arm | σ(2 s) matched | σ(6 s) matched | multiple | limit | verdict | `threshold_6s` |
|---|---|---|---|---|---|---|
| refc-xl-30k | 4.8963 | 18.3519 | **3.7481** | 3.0 | **REDERIVE** | `null` |
| refc-base-30k | 4.3359 | 16.1473 | **3.7241** | 3.0 | **REDERIVE** | `null` |

⚠️ Both σ are **re-fit on the 681 windows valid at BOTH horizons** — comparing a full-grid
σ(2 s) against a truncated σ(6 s) would be the "never compare across different windows"
defect. §5.3, verbatim: *"the ratio form does not transfer; the threshold must be re-derived,
not scaled."* **No branch of this instrument emits a scaled 6 s threshold**
(`test_no_branch_ever_emits_a_scaled_6s_threshold`).

### 4d. Ridge fit quality, for context — the σ is not a broken fit

| arm | horizon | R²_oof long | R²_oof lat |
|---|---|---|---|
| refc-xl-30k | 2 s | 0.8768 | 0.7169 |
| refc-xl-30k | 6 s | 0.8335 | 0.3450 |
| refc-base-30k | 2 s | 0.8843 | 0.7815 |
| refc-base-30k | 6 s | 0.8770 | 0.4004 |

The ridge is doing real work (it is nowhere near predicting the mean). σ is large because
the **target** is large: a 2 s endpoint at highway speed is ~28 m, and 12 % unexplained
longitudinal variance is metres. **The lateral axis collapses at 6 s (R² 0.345/0.400)** —
that is where the goal head actually fails.

### 4e. The four families (§ binding rule)

| family | reported | note |
|---|---|---|
| **LONGITUDINAL** | `sigma_long_m` per horizon (6.575 / 23.212 XL) | target-speed and distance-keeping are **n/a with reason**: this is an endpoint-capacity probe, it rolls no trajectory |
| **LATERAL** | `sigma_lat_m` per horizon (1.069 / 11.609 XL) | heading / curvature / yaw-rate **n/a with reason**: no trajectory is rolled |
| **TACTICAL** | ⭐ **the headline** — σ/ADE and σ/oracle *are* the goal/anchor-selection admissibility test | |
| **STRATEGIC** | **n/a with reason** | PhysicalAI-AV ships no map, lane graph, junction annotation or route signal |

---

## 5. ⭐ The 0-parameter floor — what actually makes this decision-relevant

§3.1's own requirement table contains a **CV goal (deployable, 0 params)** row scoring
**0.786** at N=256 against the supervised selector's 0.471 — i.e. a constant-velocity
extrapolation is a *usable* goal. So: can the trivial kinematic baselines beat a ridge on
frozen REF-C vision latents? **MEASURED on the same 881/681 windows** (`raw/cv_goal_floor.json`):

| goal source | σ(2 s) per-axis | σ/ADE | band | σ(6 s) per-axis |
|---|---|---|---|---|
| **ridge on `pooled`+`ctx`** (XL, VISION_ONLY) | **4.7104** | **9.99** | **REFUSED** | **18.3519** |
| ridge on `pooled`+`ctx` (base) | 4.5545 | 9.63 | REFUSED | 16.1473 |
| constant velocity (0 params) | 1.8712 | 3.97 | REFUSED | 13.3028 |
| go straight (0 params) | 1.7524 | 3.72 | REFUSED | 13.0658 |
| ⭐ **constant yaw rate (0 params)** | **1.1888** | **2.52** | **INCONCLUSIVE** | **11.3174** |

⛔ **These baselines are INADMISSIBLE as a deployed goal head** — they read the **ego pose
history**, the privileged channel the vision-only rule forbids at inference. They are a
**floor for interpretation**, labelled as such in the artifact. The ridge, by contrast, runs
on `pooled` + `ctx`, both VISION_ONLY, `any_echo: false`.

**Two readings fall out, and they are different findings:**

1. ⭐ **The frozen REF-C vision latents are 3.96× worse than one pose difference.** The
   REFUSED verdict is a statement about **this feature surface**, not a proof that a 6 s
   goal is unpredictable. SEL-1's *estimand* survives; its *input* does not.
2. ⚠️ **But even the best 0-parameter floor does not reach FUNDED.** σ\* = 0.8 m is below
   **every** baseline here (best: 1.1888 m ⇒ σ/ADE 2.52, the INCONCLUSIVE band). So
   "just use a kinematic goal" does not rescue SEL-1 either — a goal head would have to
   beat constant-yaw-rate by 1.49× *and* do it from vision alone.

---

## 6. Verdict and consequence

| | |
|---|---|
| **σ/ADE ≤ 1.7** | SEL-1 funded, S-T launches with it — **NOT MET** |
| **σ/ADE ≥ 3.0** | ⭐ **MET at 9.99 (CI lower bound 7.45)** ⇒ **SEL-1 is REFUSED before launch**; the work moves to `ANCHOR_GOAL` supervision (PH0 + `obstacle.offline`) |
| in between | inconclusive, run the capacity control first — not reached |

The pre-registration's own fallback is **supervision, not another cost function** (§4.1's
committed consequences: *"⛔ WORSE ⇒ SEL-1 is refuted as posed; the fallback is not another
cost — it is the **supervision** branch: `ANCHOR_GOAL` labels from the PH0/`obstacle.offline`
line"*), and
§5 above is the reason that fallback is the right one: the goal signal is not absent from
the world, it is absent from **these latents**.

⚠️ **The capacity control is not needed to reach this verdict** — the INCONCLUSIVE band was
never entered, on either arm, at either bound of the interval.

---

## 7. ⛔ SCOPE — this is the REF-C surface, NOT the frozen S-W latents §5.2 names

**Say it plainly: a REF-C number must not be read as a v6 one.**

§5.2 specifies *"frozen **S-W** latents only"*. **S-W latents have never been dumped.** What
exists banked, at the pre-registered 881×40 grid, is the **REF-C** latent dump from
2026-08-04. This result is therefore measured on REF-C-XL/base.

| | |
|---|---|
| what transfers | ⭐ **the RATIOS** — σ/ADE and σ/oracle, which is exactly why §5.2 states its outcomes as ratios. REF-C is also **the arm the 1.7 / 3.0 thresholds were derived on** (§3.1's fan is `fan_refined_refc-xl-30k.pt`), so the comparison is internally consistent. |
| what does **not** transfer | the absolute metres. They are REF-C's fan's. |
| **class** | **EXPLORATORY** for the absolutes; **MEASURED (ours)** for everything computed here |
| **tier** | ⛔ **T0-DIAGNOSTIC** — a representation-capacity probe on banked latents. **NOT a driving-performance number. No T1 capability claim may cite it.** |
| still owed | one GPU pass (~10–25 GPU-min, a deliberate training pause) to dump S-W latents and re-run this instrument unchanged. **The instrument, the target backfill and the val40 poses are now all in place**, so that pass is the *only* remaining input. |

⚠️ **What a S-W re-run could plausibly change.** S-W is trained as a world model, not as
REF-C's fan-emitter, so its latents could carry more goal information. For the verdict to
flip to FUNDED, S-W would have to reach σ ≤ 0.80 m — **5.89× better than REF-C** *and*
1.49× better than the 0-parameter kinematic floor, from vision alone. To reach even
INCONCLUSIVE it needs 3.33×. **That is the number to pre-register against before the pass
is spent**, so the S-W run is a test rather than a hope.

---

## 8. Tests

`stack/`: **`pytest -q -p no:cacheprovider`** → ⭐ **3029 passed, 17 skipped, 2 xfailed**,
exit 0, 348 s. Log: `raw/stack_pytest.txt`.

⚠️ **The count is 110 above the briefed HEAD-`bd9f49f` baseline of 2919, and that is NOT
mine — do not read it as this turn's contribution.** My change is **+5** tests
(`test_e_wc2_sigma_star.py`, 62 → 67 collected; +76 lines, all in the alignment-gate block).
The rest is **concurrent sibling work** landing in the same tree: three test files exist that
were not at `bd9f49f` — `test_bev_consumer_fov.py` (staged by a sibling), plus untracked
`test_p8_v6.py` and `test_physicalai_feature_readset.py` (**51 collected between them**).
**0 failures, 0 errors**, so nothing here regressed anything.

---

## 9. Deliverable manifest

| artifact | repo path |
|---|---|
| **This report** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-ewc2-result/EWC2_RESULT.md` |
| ⭐ **σ\* result, REF-C-XL** | `…/2026-08-16-ewc2-result/raw/ewc2_sigma_star_refc-xl-30k.json` |
| ⭐ **σ\* result, REF-C-base** | `…/2026-08-16-ewc2-result/raw/ewc2_sigma_star_refc-base-30k.json` |
| **Preconditions verified on the real dumps** (LOEO disjointness + per-axis unit + leak control) | `…/2026-08-16-ewc2-result/raw/ewc2_preconditions_verified.json` |
| **0-parameter goal floor** | `…/2026-08-16-ewc2-result/raw/cv_goal_floor.json` |
| **Alignment-gate evidence, per arm** | `…/2026-08-16-ewc2-result/raw/endpoint_backfill_controls_refc-{xl,base}-30k.json` |
| **val40 poses parity proof** (40/40 sha256 vs the committed manifest) | `…/2026-08-16-ewc2-result/raw/val40_hf_poses_verify.json` |
| Test log | `…/2026-08-16-ewc2-result/raw/stack_pytest.txt` |
| Precondition run stdout | `…/2026-08-16-ewc2-result/raw/preconditions_stdout.txt` |
| **Code — HF range-read poses puller** (255× cheaper than the cache) | `…/2026-08-16-ewc2-result/code/hf_poses_pull.py` |
| **Code — precondition verifier** | `…/2026-08-16-ewc2-result/code/verify_ewc2_preconditions.py` |
| **Code — 0-parameter goal floor** | `…/2026-08-16-ewc2-result/code/cv_goal_floor.py` |
| ⭐ **Backfilled latent dumps** (the reusable surface: latents + 2 s/6 s GT endpoints) | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-04-lambda-findability/raw/latents_refc-{xl,base}-30k-ep.pt` |
| **Hardened alignment gate** | `stack/scripts/refc_dump_latents.py` (`ENDPOINT_ULP_TOL`, `_row_ulps`, `endpoint_agreement`) |
| **Its 5 tests** | `stack/tests/test_e_wc2_sigma_star.py` |

**Off-repo state:** a poses-only val40 view (40 `ep_*.pt` stubs with zero-placeholder frames)
in this session's scratchpad at `…/scratchpad/val40hf/physicalai-val-0c5f7dac3b11/`.
**Deliberately not committed** — it is a 4.70 GB cache's derivative and `code/hf_poses_pull.py`
regenerates it in 81 s. **Nothing was written to any episode cache**, so
`physicalai-val-0c5f7dac3b11` and skip-hash `f09e44db` are untouched by construction.
⛔ **Thor was not contacted: no ssh, no disk read, no job.** Nothing was pushed anywhere.

---

## 10. ⛔ ESCALATION — integration, not a note in a README

1. ⭐ **`MODEL_REGISTRY.md` / `V6F_PLANNER_DESIGN.md` §5.2 need the outcome recorded.** The
   pre-registration's committed consequence has now fired: **SEL-1 is REFUSED before
   launch, and S-T must not launch with it.** A design doc that still reads "E-WC2 should
   run before S-T is launched" is stale as of this turn. **This is a doc edit I did not make
   — `V6F_PLANNER_DESIGN.md` and the registry belong to the orchestrator/PI.**
2. ⭐ **The fallback branch is now the live path: `ANCHOR_GOAL` supervision from PH0 +
   `obstacle.offline`.** §5 says why it is the right one rather than a consolation:
   the goal signal exists (a 0-param baseline finds 3.96× more of it than the latents do),
   it is simply not in the frozen REF-C representation.
3. **Pre-register the S-W re-run before spending the pause.** The instrument, the endpoint
   backfill and the val40 poses are all in place; the remaining input is one ~10–25 GPU-min
   dump. The thresholds it must beat are **σ ≤ 0.80 m (FUNDED)** and **σ ≤ 1.41 m
   (leave REFUSED)** — 5.89× and 3.33× better than REF-C respectively.
4. **`RETRACTION_LOG.md` gets the §2 root-cause class** — *an exactness check whose
   exactness is a property of the machine, not of the claim*. Append-only and the
   orchestrator's file; the text is in §2 ready to lift.
5. ⚠️ **The C69 correction is confirmed and can be closed out**: the REF-C route needed no
   GPU, and its one missing input — the val40 poses — turned out to be reachable **without
   Thor at all**. The "single copy on Thor" reading of the val40 cache is itself
   incomplete: the 256 px cache is mirrored on HF and is **range-readable**, which makes a
   poses-only view an 18 MB, 81-second operation from any box.
