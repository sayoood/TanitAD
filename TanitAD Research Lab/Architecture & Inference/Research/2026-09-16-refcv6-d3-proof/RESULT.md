# D3 proof package — RESULT

**Run 2026-09-16 (Europe/Berlin) on the dev-box RTX 4060.** Pre-registration:
`PREREG_D3_PROOF.md`, frozen (`git add`) before the first number was read.
Authority: `Project Steering/REFCV6_CLARIFICATION.md` §2.4 + `SPEC_REFCV6_V2.md` §8.

> ## ⭐ THE ONE LINE
> **NO — refcv5-v2's planner does NOT demonstrably use the scene as a scene.** It uses the
> image *globally* (blanking every frame costs **+1.04 m** ADE) and it uses the *ego block*
> overwhelmingly (withholding it costs **+7.51 m**), but **greying the LEAD VEHICLE out of
> every frame moves its minimum time gap by −0.0005 s — 0.09× the model's own
> inference-seed noise floor, and not separated.** T-B fails all three registered
> conditions. The planner reads the image the way a texture is read, not the way a lead is
> read.
>
> **The other half — extraction — is thin but real:** a frozen-token box probe beats every
> zero-information control (AP@2 m 0.0631 vs shuffled 0.0304 / pixel 0.0278 / prior 0.0221,
> separated, 7× the training-seed floor, on 2 seeds) — **but below its +0.05 bar, with
> recall 0.000 at precision 0.50, a ~6 m lead-range error, and two registered controls
> failing: stride-16 does NOT beat stride-8-by-20, and a MIRRORED token address does NOT
> lose.**

### Verdict table

| test | registered bar | verdict |
|---|---|---|
| T-A `frames_blind` | gate, controls binding | ✅ **PASS** (controls exact) — the image is worth **+1.04 m** ADE |
| T-A `ego_zero` | gate, controls binding | ✅ **PASS** (controls exact) — the ego block is worth **+7.51 m** |
| T-G attention | reported with its control | ✅ control exact (uniform = 1.0000000056); lead concentration **1.92×**, separated |
| T-H decision probe | gap coefficient separated, human's sign | ⛔ **FAIL** — and the **human's own** gap coefficient is not separated either (a pre-registration defect, recorded) |
| ⭐ **T-B lead mask** | separated **AND** ≥ 3× seed floor **AND** unsafe | ⛔⛔ **FAIL at the first condition**; the effect is **0.09×** the seed floor |
| Rung 2 **B1** box AP | ≥ +0.05 over max control, separated, ≥ 3× floor, 2 seeds | ⛔ **FAIL on magnitude** (+0.033); separation, floor and seeds pass |
| Rung 2 **B2** lead AUC | ≥ +0.10 over a within-clip shuffle | ⛔ **FAIL** (+0.051) |
| Rung 2 **B3** s16 > s32 | separated | ⛔ **FAIL** — not separated on either detector |
| Rung 2 mirror control | must LOSE | ⛔ **FAIL** — it does not |

---

## 0. What was run, and on what

| | |
|---|---|
| model | refcv5-v2 `ckpt.pt` **step 40,284**, `C:/Users/Admin/refcv5v2_final/` |
| harness | `taniteval/tools/refcv3_arm.py` (REF-C T1 adapter, four families) |
| code tree | **`C:/Users/Admin/refcv5cmp/repo`** — see §1 |
| grid | 2 s, K = 4, `--window-stride 5`, `--action-units steer` |
| n | **4,823 windows / 141 episodes**, tier **T1 (self-action OPEN loop)** |
| clip ids | sha12 = `sha256(clip_id)[:12]`, everywhere |
| **GPU cost** | **4.46 GPU-h banked** (sum of every kept job's own `wallclock_s` / `wall_s`) **+ ≈ 0.9 h discarded** (a 5/141 random-mask roll withdrawn for a control-design fix, a 37/141 duplicated `ego_zero` roll from §9(b), and 2 aborted 4-epoch box arms) **+ ≈ 0.13 h** for T-G. **≈ 5.5 of the 12 GPU-h budget.** Rolls: `blind` 31.9 min · `ego_zero` 32.6 · `leadmask` 35.5 · `randmask` 32.3 · provenance + zero-mask 1.2. Box arms: 10.1–35.9 min each |

---

## 1. ⛔ PROVENANCE GATE — and a defect it caught before it cost anything

`stack/tanitad/refs/refc.py` differs from the branch tip by **321 content lines**
(`refc_v3.py` 33, `refc_v3_train.py` 1,057); `refcv3_arm.py` and `v2_dataset.py` are
content-identical in all three trees (line endings only). Rolling on the tip would have
silently un-paired every ablation from the banked baseline.

**MEASURED, `code/provgate.py`:** rolled on `C:/Users/Admin/refcv5cmp/repo` with the
banked call structure, `os` reproduces the banked `refcv5-v2` dump **BIT-EXACTLY**
(`exact_equal = 1.000000` on `g`, `os`, `ha`, `ha0`, `ha0_ext`, ep000–ep001).

⚠️ **The defect the gate caught.** A first attempt used `--no-navshuf --no-navzero` to
make the fed batch 1 row instead of 3 — 2× cheaper. `g`/`ha`/`ha0`/`ha0_ext` stayed
bit-identical, but **`os` moved by max 1.18 m / mean 0.065 m**: the ddim sampler's noise
draw follows the batch shape. Every arm in this package therefore uses the banked call
structure verbatim. ⭐ **Read this beside the "cheap" flag next time: a knob that does not
touch the model can still change its answer by 20 % of the arm's own ADE.**

---

## 2. ⭐ THE INFERENCE-SEED FLOOR — measured first, because everything is read against it

refcv5-v2 runs `--sampler ddim`, which **draws noise at eval** (`refc.py:1926`). Paired
`full(seed 0) − full(seed 1)`, both banked, episode-cluster bootstrap over clips, 2,000
resamples — **MEASURED, T1, n = 4,823 windows / 141 episodes**:

| metric | seed-0 minus seed-1 | 95 % CI | separated |
|---|---|---|---|
| ADE | **0.0000 m** | [−0.0025, +0.0025] | no |
| min headway | **+0.0242 m** | [+0.0016, +0.0523] | ⚠️ **YES** |
| **min time gap** | **+0.0054 s** | [−0.0007, +0.0117] | no |
| min TTC | **+0.0941 s** | [−0.0171, +0.2157] | no |

⛔ **Pure noise reseeding already produces a SEPARATED headway difference.** That is the
whole reason the pre-registration demands ≥ 3× the floor and not merely "separated".
**The registered floor for minimum time gap is 0.0054 s; 3× it is 0.0162 s.**

Raw: `raw/seedfloor.json`.

---

## 3. T-A — the VOID gate

### 3.1 `frames_blind` (every observed frame → the window's own scalar mean)

**MEASURED, T1, n = 4,823 windows / 141 episodes**, paired episode-cluster bootstrap.

| arm | ADE (m) | 95 % CI |
|---|---|---|
| `os` FULL (banked) | **0.3079** | [0.2795, 0.3390] |
| `os` **frames_blind** | **1.3455** | [1.2887, 1.4014] |
| `ha0` (constant velocity) | 0.6723 | [0.6007, 0.7469] |
| `ha0_ext` (the echo control) | 0.2874 | [0.2649, 0.3137] |
| `oracle_sel` FULL | 0.2303 | [0.2140, 0.2492] |
| `oracle_sel` frames_blind | 0.5821 | [0.5366, 0.6324] |

**paired `blind − full` ADE = +1.0376 m [+0.9760, +1.0999], SEPARATED.**
Distance keeping: min headway **−0.5524 m [−0.9708, −0.1955], SEPARATED** (n 806);
min time gap +0.0091 s, not separated; min TTC +0.7694 s, not separated. ⚠️ The
scoreable denominator itself falls **1,300 → 806** windows — the blind plan leaves the
lead corridor on a third of them, which is part of the answer, not a missing number.

**Registered controls — PASS, EXACTLY:**

| control | known value | MEASURED |
|---|---|---|
| `ha`, `ha0`, `ha0_ext`, `g` under `frames_blind` | BIT-IDENTICAL | **max \|Δ\| = 0.000000e+00 over ALL 141 episodes / 4,823 windows** |
| `paired(os, os)` on the banked dump against itself | EXACTLY 0 | **0.0000 on ADE, headway, time gap and TTC, with 0-width CIs** |

⇒ **The instruments CAN fail an image-blind arm.** The blind arm scores **worse than
`ha0`** (doing nothing) and **4.7× worse than `ha0_ext`** (the echo). The panel is not
void, and **the image path is load-bearing at 1.04 m of ADE.**

⭐ Reference: on **refcv4b** blanking the frames moved ADE 0.2975 → 1.0491 (Δ +0.752).
refcv5-v2 moves **+1.038** — it depends on the image **MORE**, not less.

### 3.2 `ego_zero` (`ego_state[:,4] = 0` AND `v0 = None` at the core)

**MEASURED, T1, n = 4,823 windows / 141 episodes.**

| arm | ADE (m) | 95 % CI |
|---|---|---|
| `os` **ego_zero** | **7.8224** | [6.7240, 9.0482] |
| `os` FULL | 0.3079 | [0.2795, 0.3390] |
| `os_navshuf` ego_zero | 7.8315 | [6.7521, 9.0475] |
| `os_navzero` ego_zero | 7.8022 | [6.7167, 9.0096] |
| `oracle_sel` ego_zero | 7.0566 | [6.0932, 8.1129] |
| `ha` / `ha0` / `ha0_ext` | **0.2996 / 0.6723 / 0.2874 — UNCHANGED** | identical to the FULL run |

**Registered control — PASS:** the model-free arms keep the MEASURED v0 and read exactly
their FULL-run values, so the 25.4× collapse is the model's, not the corpus's.

⭐ **The ego block is ~7× more load-bearing than the whole image.** Withholding the ego
state costs **+7.51 m** of ADE; blanking every pixel costs **+1.04 m**. A model whose plan
falls apart without its own speed and acceleration, and barely notices the lead vehicle
(§6), is an ego-extrapolator with a vision-shaped texture prior.

⚠️ **Provenance of this row.** The first `ego_zero` roll finished cleanly and its record is
banked (`raw/t_a_egozero_run1_fourfamilies.json`), but its DUMP was clobbered — see §9.
The re-roll of 2026-09-17 is the primary artifact and reproduces these ADEs (same
`--infer-seed 0`, deterministic; verified in §9).

---

## 4. T-G — attention attribution

**MEASURED, T1, n = 600 windows with a lead ≤ 30 m / 43 clips** (mean gap 16.96 m).
`refc.CrossAttnLayer.cross` (`refc.py:1289`, called at `:1315` with `need_weights=False`)
was wrapped so the same call is made with `need_weights=True`; the module's own attention
output is returned unchanged. Token grid **8 × 20 = 160**; the lead covers **2.98 columns**
on average (area share 0.149). Raw: `raw/t_g.json`.

`concentration = (mass on the lead's token columns) / (their area share)`, on the
**selected** anchor's row:

| decoder layer | lead | 95 % CI | random equal-width band | lead − random (paired) | separated |
|---|---|---|---|---|---|
| 0 | **1.5988** | [1.4522, 1.7434] | 0.8061 | **+0.7935** [+0.6042, +0.9869] | ✅ |
| 1 | **1.2864** | [1.1860, 1.3895] | 0.8669 | **+0.4196** [+0.2692, +0.5735] | ✅ |
| **2** | **1.9196** | [1.7827, 2.0472] | 0.7796 | **+1.1458** [+0.9879, +1.2814] | ✅ |
| 3 | **1.5726** | [1.4323, 1.7028] | 0.8228 | **+0.7516** [+0.5903, +0.9121] | ✅ |

**Registered control — PASS, EXACTLY:** a uniform attention row, scored by the same
function, reads **1.0000000056** (max deviation from 1 over all 600 windows ×4 layers =
**1.19 × 10⁻⁷**, i.e. float32 rounding). Every attention row sums to **1.0**.

⇒ **The selected anchor DOES address the lead: up to 1.92× its area share, 2.46× the
random band, separated at every layer.**

⭐⭐ **And this is the sharpest result in the package, because §6 says the same masking
changes nothing.** The decoder looks *at* the lead and does not act *on* it. An attention
map is therefore **not** evidence of use — which is exactly what §2.2 of the clarification
warned and what this package was built to test.

---

## 5. T-H — the decision probe

**MEASURED, T1, n = 1,487 lead windows / 78 episodes.** OLS of the planned 0–2 s
longitudinal acceleration on (gap, closing rate, v0, a0), episode-cluster bootstrap over
clips, 2,000 resamples. Raw: `raw/t_h.json`.

| arm | gap (m/s² per m) | closing rate (per m/s) | a0 | mean plan accel |
|---|---|---|---|---|
| **`os` seed 0** | **−0.000320** [−0.00131, +0.00061] ⛔ not sep. | **−0.00382** [−0.01089, −0.00069] ✅ sep. | +0.7789 [0.6988, 0.8475] | +0.128 m/s² |
| `os` seed 1 | −0.000116 [−0.00103, +0.00089] not sep. | −0.00405 [−0.01149, −0.00088] sep. | +0.7751 | +0.130 |
| **human (GT)** | **+0.000623** [−0.00076, +0.00208] ⛔ not sep. | **−0.01377** [−0.03353, −0.00685] ✅ sep. | +0.8631 | +0.086 |
| `ha0_ext` (control) | **+8.2e−18** | −9.0e−18 | **+1.0000000000000** | +0.033 |
| `ha0` (control) | +2.0e−09 | +1.1e−08 | −2.1e−08 | 0.0000 |

**Registered controls — PASS, EXACTLY.** `ha0_ext` plans `accel ≡ a0`: its gap
coefficient is **8.2 × 10⁻¹⁸** (machine zero) and its a0 coefficient is **exactly 1**.
`ha0` plans `accel ≡ 0`: every coefficient ≤ 1e−8.

**VERDICT AGAINST THE REGISTERED BAR: ⛔ FAIL.** The bar was *"the model's gap
coefficient is separated from 0 and has the same sign as the human's"*. The model's gap
coefficient is **not separated**, and its sign is opposite the human's point estimate.

⚠️ **But the bar itself does not survive contact with its own reference: the HUMAN's gap
coefficient is not separated either** (n 1,487 / 78 clips). On this corpus, at this
horizon, *gap* is not what either driver's 0–2 s acceleration responds to. Registering a
bar the reference cannot clear is a pre-registration defect, and it is recorded here
rather than quietly re-specified.

⭐ **What IS separated, for both, is the CLOSING RATE** — and with the *safe* sign
(faster approach ⇒ lower planned acceleration): the model **−0.00382**, the human
**−0.01377**. **The model has the human's sign at 0.28× the magnitude.** This was not the
registered gate, so it is reported as a finding, not as a pass.

⚠️ **Seed sensitivity:** the gap coefficient moves −0.000320 → −0.000116 between two
inference seeds — **64 % of its own point estimate**. Any future bar on this quantity
must be registered against that floor.

---

## 6. ⭐ T-B — THE LEAD MASK, the deciding test

### 6.1 What the intervention actually did

The lead cuboid (≤ 30 m, the lead block's own `lead_track_id`) was projected through each
clip's own front-wide extrinsics into **all 24 sub-frames** of every window (8 provider
rows × 3 D-015 channel groups, each at its own raw frame) and filled with the frame mean.

| | lead arm | random arm |
|---|---|---|
| windows masked | **885** of 4,823 (18.3 %) | **738** |
| sub-frames masked | 21,124 | 17,626 |
| mean mask area | 4,774 px (median 1,551, p95 22,780) | **3,257.31 px — EXACTLY equal to the lead arm's on the same sub-frames** |
| masked pixel fraction | 2.91 % | 1.99 % |
| `rand_fallback` | — | **147** windows had no off-agent placement at the origin frame |
| `rand_clamped` | — | 226 of 17,626 sub-frames (1.3 %) |
| `rand_hits_agent_offorigin` | — | 3,219 of 17,626 (18.3 %) — the per-window shift overlaps *some* agent in a non-origin sub-frame |

⭐ **The mask was LOOKED AT before it was scored** (`RETR-2026-09-13-SAM3MAP-A2-GHOST-GROUND`:
a z-derived mask once blanked the road in front of every vehicle and only a picture caught
it). Three rendered strips are banked at `raw/mask_preview/` — original / lead / random —
and the lead box lands on the vehicle, not on the road in front of it. The cuboid is built
**ground-standing** (bottom at rig z = 0, top at an ESTIMATED per-class height), which is
that retraction's correction: the agent join carries no z at all.

**Registered controls — PASS, EXACTLY:**

| control | known value | MEASURED |
|---|---|---|
| zero-area mask (full code path, zero fill) | plan BIT-IDENTICAL | **`exact_equal = 1.000000` on `g`, `os`, `ha`, `ha0`, `ha0_ext`**, both probe episodes, with 816 sub-frames processed |
| model-free arms under the lead mask | BIT-IDENTICAL | **max \|Δ\| = 0 over all 141 episodes** |
| the mask reached the model | the masked windows, and ONLY those, move | **exactly 885 of 4,823 plans changed — the mask summary's own count — and 3,938 are bit-identical** |
| per-window magnitude | — | mean \|Δ\| **0.1749 m** on changed windows, p50 0.055 · p90 0.574 · p99 1.339 · max 3.514 |

⇒ The intervention is real and it moves individual plans by metres. The question is
whether it moves them in the **direction that matters**, more than an equal-area patch of
empty road does.

### 6.2 ⛔ THE VERDICT: FAIL

**`leadmask − randmask`, restricted to the 738 windows BOTH arms masked** (60 episodes),
paired episode-cluster bootstrap over clips, 2,000 resamples. Negative = the lead-masked
plan approaches the lead MORE, the unsafe direction.

| read | Δ (lead − random) | 95 % CI | separated? | vs the seed floor |
|---|---|---|---|---|
| **minimum time gap** ⭐ | **−0.0005 s** | [−0.0084, +0.0076] | ⛔ **NO** | **0.09×** the 0.0054 s floor (the bar is ≥ 3×) |
| minimum headway | −0.0127 m | [−0.0408, +0.0100] | NO | 0.53× the 0.0242 m floor |
| minimum TTC | +0.0620 s | [−0.2144, +0.3743] | NO | 0.66× the 0.0941 s floor, and the **SAFE** sign |
| ADE (touched windows) | +0.0016 m | [−0.0072, +0.0095] | NO | — |

**Closing windows** (n = 306 / 39 episodes):

| read | leadmask | randmask | human | paired Δ | separated |
|---|---|---|---|---|---|
| planned 0–2 s accel (m/s²) | −0.0450 | −0.0578 | **−0.1775** | +0.0127 [−0.0078, +0.0344] | NO |
| speed error vs the human (m/s) | 0.6063 | 0.5845 | — | +0.0218 [−0.0106, +0.0576] | NO |

**Against the pre-registered PASS — all three conditions fail at the first:**

1. separated — **NO**;
2. ≥ 3× the inference-seed floor — **NO (0.09×)**;
3. unsafe direction — the point estimates lean unsafe on time gap, headway and planned
   deceleration, but none is separated and TTC leans the other way.

⇒ **REGISTERED FAIL: "a lead mask ≈ a random mask. The planner is then not reading the
lead, whatever a probe decodes."**

### 6.3 Two readings that survive the failure

- **Secondary (lead mask vs the UNMASKED baseline, 885 touched windows / 61 episodes):**
  min time gap −0.0007 s [−0.0179, +0.0123], min headway +0.0025 m [−0.0313, +0.0426],
  min TTC +0.2633 s [−0.0064, +0.5606], ADE −0.0017 m — **none separated**. Removing the
  lead vehicle entirely from the input costs the plan *less than reseeding the sampler.*
- ⭐ **A separate finding, not a T-B read:** in closing windows the plan decelerates at
  **−0.045 m/s² against the human's −0.177 m/s² — 25 % of the human's braking.** The
  longitudinal weakness this programme has been chasing is visible here as a *rate*, not
  only as an error.

---

## 7. RUNG 2 — the box probe on the cached frozen tokens

**MEASURED.** Frozen refcv5-v2 tokens, `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/`
(N = 27,664 stacked rows), target = the B1 EVAL agent join (md5
`3ddb42ecbd3926066795a94587af2aed`), **26,250 rows carry a join row**, mean **2.15**
vehicles per row in 0–30 m, lead prevalence **0.1981**. Clip-disjoint content-blind split
(`p4_bev_head`'s own rule): **train 16,082 rows / 86 clips · val 3,453 / 18 · test 6,715 /
35**, **15,327 GT boxes** in the test split. 8 arms × 10 epochs; wall 605–2,157 s per arm.
Raw: `raw/box_panel.json`.

⚠️ **Arm tags.** The pre-registration names the s16 arms `s16` / `s16_s1` / `s16_shuffled` /
`s16_mirror`; the run tags them `main_s16` / `main_s16_s1` / `shuf_s16` / `mirror_s16`. Same
eight arms, same levers — only the label differs, and it is recorded here rather than left
for a reader to reconcile.

⚠️ **TWO DETECTORS, BOTH REPORTED.** The pre-registration asked for "centre heatmap + DETR
slots". At this budget the **slots** decode better than the heatmap, and reporting only the
winner would be a selection the pre-registration does not license, so **every bar below is
applied to both**. (MEASURED en route: at 4 epochs without sub-cell offsets the slots scored
AP@2m **0.0148** — *below* the zero-image `prior` **0.0221** — which is DETR's known
convergence behaviour, not a statement about the tokens. 10 epochs + CenterNet sub-cell
offsets moved them to 0.0631. The 4-epoch arms were discarded, not reported.)

### 7.1 The panel (AP@2 m, 0–30 m, centre matching, clip-cluster bootstrap, 1,000 draws)

| arm | tokens | **slots AP@2 m** | slots AP@4 m | heatmap AP@2 m | lead AUC | − within-clip shuffle | lead range MAE |
|---|---|---|---|---|---|---|---|
| `main_s32` | s32 8×20 | **0.0631** | 0.1990 | 0.0324 | 0.8133 | **+0.0506** | 5.60 m |
| `main_s16` | s16 16×40 | **0.0625** | 0.2080 | 0.0361 | 0.8332 | **+0.0591** | 6.30 m |
| `main_s32_s1` (seed 1) | s32 | 0.0586 | 0.1844 | 0.0386 | 0.7970 | +0.0656 | 6.77 m |
| `main_s16_s1` (seed 1) | s16 | 0.0571 | 0.1962 | 0.0387 | 0.8144 | +0.0479 | 6.83 m |
| ⛔ `mirror_s16` | s16, flipped | **0.0558** | 0.2090 | 0.0388 | 0.8251 | +0.0557 | 6.30 m |
| `shuf_s32` | s32, targets permuted | 0.0304 | 0.1440 | 0.0262 | 0.6745 | +0.0351 | — |
| `shuf_s16` | s16, targets permuted | 0.0188 | 0.1056 | 0.0141 | **0.5070** | −0.0216 | 9.20 m |
| `pixel` | pix64 raw pixels | 0.0278 | 0.1594 | 0.0299 | **0.5597** | −0.0127 | 9.36 m |
| `prior` (analytic) | none | **0.0221** | 0.0596 | — | — | — | — |
| `const` (analytic) | none | — | — | — | **0.500000** | — | — |

**Controls against their KNOWN values:**

| control | known value | MEASURED | |
|---|---|---|---|
| `const` lead AUC | **EXACTLY 0.5** | **0.500000** | ✅ |
| `const` lead accuracy | **EXACTLY the prevalence** | **0.220551 = the test prevalence** | ✅ |
| `shuffled` ≤ `prior` + 0.02 | ≤ 0.0421 | s32 **0.0304**, s16 **0.0188** | ✅ |
| `pixel` floor | reported | **0.0278** | ✅ |
| training-seed floor | reported | s32 **0.0045**, s16 **0.0054** (AP@2 m) | ✅ |
| `shuf_s16` / `pixel` lead AUC | ≈ chance | **0.5070 / 0.5597** | ✅ |
| ⛔ **`mirror_s16` MUST LOSE** | a mirrored address must cost AP | `main_s16 − mirror_s16` = **+0.0067 [−0.0048, +0.0255], NOT separated** | ⛔ **FAIL** |
| ⚠️ `shuf_s32` lead AUC | ≈ chance | **0.6745** | ⛔ **FAIL** — the s32 shuffled arm's lead head is not at chance, so the s32 lead-AUC row is weakly controlled (s16's control is clean) |

### 7.2 Verdict against the pre-registered bars

| bar | requirement | MEASURED | verdict |
|---|---|---|---|
| **B1** | AP@2 m − max(shuffled, prior, pixel) **≥ +0.05**, separated, ≥ 3× seed floor, 2 seeds | slots s32: **+0.0327** vs `shuf_s32` [+0.0051, +0.0706] **separated**, **7.3×** the 0.0045 seed floor, and seed 1 agrees (+0.0282, separated). Same picture on s16 (+0.0437 vs `shuf_s16`). **Heatmap: +0.0063 (s32), not separated.** | ⛔ **FAIL on magnitude** (+0.033…+0.044 < +0.05); passes separation, the 3× floor and 2 seeds on the slots detector; fails outright on the heatmap |
| **B2** | lead AUC − within-clip shuffle **≥ 0.10** | **+0.0506** (s32) · **+0.0591** (s16) · seeds +0.0656 / +0.0479 | ⛔ **FAIL** (half the bar) |
| **B3** ⭐ | **s16 beats s32**, separated | slots **−0.0007** [−0.0223, +0.0277] · heatmap **+0.0037** [−0.0097, +0.0119] | ⛔ **FAIL — NOT separated on either detector** |
| **B4** (reported) | AP@1 m, AP@4 m, recall at precision 0.50, closest-in-path range MAE | AP@1 m **0.0072** · AP@4 m **0.199** · **recall@precision 0.50 = 0.000 for EVERY arm** (precision never reaches 0.50) · range MAE **5.60–6.83 m** | registered usefulness (recall ≥ 0.50 at precision ≥ 0.50; range MAE ≤ 2 m) ⛔ **missed by a wide margin** |

### 7.3 ⭐ What this means for the spec, stated plainly

1. **There IS box information in the frozen trunk** — separated above the shuffled target,
   the per-cell prior and the raw-pixel floor, at 7× the training-seed floor, on two seeds.
   It is **not zero**, and the *"≈ raw pixels"* reading of §6 of the clarification is too
   harsh at AP@2 m (0.0631 vs 0.0278, separated).
2. **But it is thin**: +0.033 against the best control, against a +0.05 bar; no arm reaches
   precision 0.50 at any recall; the closest-in-path range is wrong by **~6 m**, three times
   the registered ≤ 2 m.
3. ⛔⛔ **THE SPEC'S STRIDE-16 PREMISE IS NOT SUPPORTED BY THIS PROBE.**
   `SPEC_REFCV6_V2.md` §2 justifies hanging perception on stride-16 with *"an oracle on 8×20
   tops out at AP 0.3341 vs 0.4713 on 16×40"*. That reference is an **ORACLE on a BUILT
   feature map with a known address** (`…/2026-09-07-wpa-readout-localisation/ORACLE_RECHECK.md`,
   corrected ladder **16×40 0.4762 · 8×20 0.3341**, seed-1 0.4651 — ⚠️ the spec's 0.4713 is
   not the banked value; **the primary source governs**), not a probe on this frozen trunk,
   and it uses that package's tie-group AP, not this one's centre-matched AP. **Measured
   here, on the trunk refcv6 would actually inherit, 16×40 does NOT beat 8×20 on either
   detector.** The oracle ladder says *a perfect reader would prefer 16×40*; it does not say
   *this trunk's 16×40 tokens carry more*. Those are different claims and only the second
   one licenses the design choice.
4. ⛔ **The mirrored-address control did not lose.** AP@2 m is invariant to flipping the
   token grid left–right — the same invariance class as `R-2026-09-08-wpa-mirror`. Whatever
   the head is decoding at 2 m, it is **not** reading an azimuth address from the tokens;
   range and clip-level context can produce this AP without left-right sense. ⭐ **Any
   refcv6 arm that hangs a BOX head on these tokens must carry this control, and must show
   it losing, before its AP is quoted as localisation.**

---

## 7.4 ⭐ The question §2.4 asked this package to settle — read vs supervise

⛔ §2.4, verbatim: *"This package cannot prove refcv6 works. It decides whether refcv6 should
bet on **reading** the trunk (arms c/e) or on **supervising** it (arm d), before ≥ 65 pod
GPU-hours are spent."*

**What the measured pattern points at.** The planner already **attends** to the lead (T-G:
1.92×, separated at every decoder layer) and the trunk already **carries** box information
(Rung 2: separated above every zero-information control). Yet masking the lead out of every
frame changes the plan no more than masking empty road (T-B: 0.09× the noise floor), and the
plan's 0–2 s acceleration does not respond to the gap (T-H). ⇒ **A better READ of the same
trunk — arms b (position encoding) and c (waypoint-indexed sampling) — is not what this
evidence points at: the read is already there and is already unused.** What is missing is a
reason for the planner to *act* on it: a **supervised** target that puts the lead in the loss
(arm d), and/or the RL / ≥GT lever of §7 of the clarification.

⛔ This is a DIRECTION, not a proof. The five-arm experiment of §2.3 is still what settles it,
and arm c remains the cheap knockout that would refute this reading if it separated.

---

## 8. What was NOT run — stated, not omitted

| item | why |
|---|---|
| **T-C road-edge mask** | needs the SAM3 semantic maps of the eval clips, which are **not on this box** (D4 unanswered; Thor is still producing). NOT RUN. |
| the **map** panel (9-class SAM3) | same reason. NOT RUN. |
| **occupancy on decoder outputs** | needs a banked decoder-output dump; out of this budget. NOT RUN. |
| T-D scene swap · T-E frozen motion · T-F pasted agent | out of this budget. NOT RUN. |
| gradient-conflict detector · tiny-rig joint arms (rung 2/3 of the extraction ladder) | out of this budget; they are §2.4's days 4–5. NOT RUN. |
| a T-B **inference-seed replicate of the mask arms** | the registered floor was taken from the banked FULL seed-0/seed-1 pair (§2), which is the pre-registration's own definition. A mask-arm replicate would have cost 2 further full rolls and could not change a verdict that failed at 0.09× the floor. NOT RUN, deliberately. |

---

## 9. ⛔ TWO OPERATIONAL DEFECTS OF THIS SESSION, recorded because they cost GPU hours

**(a) `cmd | tee log` hides a dead roll.** `roll.sh`/`maskroll.sh` piped into `tee`, so the
pipeline's exit status was **tee's**. When a roll was stopped deliberately, the queue
printed its own `QUEUE_TB_DONE` marker and the next stage started on a partial dump. This
is the `never-chain-the-lander-behind-a-pipe` class. **Fixed:** `set -o pipefail` in every
queue and roll script in `code/`.

**(b) ⭐ EDITING A SHELL SCRIPT WHILE BASH IS EXECUTING IT.** `set -o pipefail` was added to
`roll.sh` and `queue2.sh` *while both were running*. **bash re-reads a script by BYTE
OFFSET**, so the insertion shifted every later byte: the running `roll.sh` executed a
fragment of its own command line (`le-sel: command not found`) and the running `queue2.sh`
**re-executed a whole block** — a second `ego_zero` roll that `rm -rf`'d the first roll's
dump while the first roll's JSON survived. Cost: ~40 GPU-minutes and one clobbered dump.
**Fixed:** the re-run used a fresh script (`/c/Users/Admin/d3_out/queue_final.sh`) carrying
the rule in its own header; nothing under `code/` was edited while a queue was live.
⭐ **The repair is verifiable rather than argued:** the clean re-roll reproduces run 1's
`ego_zero` panel **exactly** — `os` 7.8224 [6.7240, 9.0482] and all six other arms
identical to 4 dp.

---

## 10. Deliverable manifest

⛔ **Anything marked ONE PLACE exists nowhere else.** The dumps are large and stay off the
repo; everything needed to *read* the verdicts is in the package.

| artifact | where | note |
|---|---|---|
| `PREREG_D3_PROOF.md` | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-16-refcv6-d3-proof/` (worktree `wt-d3-20260916`, staged) | frozen before any number |
| `RESULT.md` | same | this file |
| `code/` (16 files) | same | `leadmask.py` · `run_mask_roll.py` · `mask_preview.py` · `t_b_paired.py` · `t_g_attention.py` · `t_h_decision_probe.py` · `p_box_head.py` · `p_box_eval.py` · `provgate.py` · `scrub_sha12.py` · `roll.sh` · `maskroll.sh` · `queue_tb.sh` · `queue2.sh` · `queue3_box.sh` · `queue4_box.sh` |
| `raw/*.json`, `raw/mask_preview/*.png`, `raw/box_runs/<tag>/config.json` | same | every number quoted above, plus each box arm's exact configuration |
| the 5 refcv3_arm **DUMPS** (`blind`, `leadmask`, `randmask`, `egozero`, `zeromask`) | **devbox:`C:/Users/Admin/d3_out/*_dump/`** | ⛔ **ONE PLACE.** ~40 MB each; re-creatable from `code/roll.sh` / `code/maskroll.sh` at `--infer-seed 0` (MEASURED bit-reproducible) |
| the 8 box-head runs (`config.json`, `test_pred.npz`) | **devbox:`C:/Users/Admin/d3_out/box/<tag>/`** | ⛔ **ONE PLACE.** re-creatable from `code/queue4_box.sh` |
| roll logs | **devbox:`C:/Users/Admin/d3_out/*.log`** | ⛔ ONE PLACE |
| the banked baseline it is all paired against | `devbox:C:/Users/Admin/refcv5cmp/out/refcv5-v2_dump` + `…-seed1_dump` | pre-existing; also banked in-repo as `…/2026-09-07-refcv5-v2-comparison/raw/` |
| the git worktree | `worktree:wt-d3-20260916` (detached at `9a782fa`) | **48 files staged, NOT committed, NOT pushed** |

⛔ **INTEGRATION — the branch moved under this package.** The worktree was cut from
`agent/arch-inf-20260803` at **`9a782fa`**; while this ran, another agent advanced that
branch to **`837c308`**. `9a782fa` is now an ancestor, so **whoever lands these 48 files must
land them on the current tip, not on `9a782fa`.** Nothing here touches a shared file — the
package is 48 NEW paths under one new directory — so the landing is additive.

**Clip-id hygiene:** `code/scrub_sha12.py` rewrote **141 clip ids per paired record** to
sha12 and a second pass reads **0 left** in every banked file.
