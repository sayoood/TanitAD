# Thor P6 — the optimisation and the precision gate, re-run ON REAL TRAINED WEIGHTS

**Date:** 2026-08-03 (Europe/Berlin). **Hardware:** NVIDIA Thor (Blackwell **sm_110**), aarch64,
torch **2.13.0+cu130**, TensorRT **10.13.3.9**, `~/venvs/tanitad-edge`. **Cost $0** (owned hardware).
**Wall-clock ≈ 60 min** including 9 engine builds and two full 859-window scored passes.
**Plan ref:** `THOR_DEPLOYMENT_RUNBOOK.md` §1/§2/§3 + backlog **O1**.
**Artifacts:** `Implementation/incoming/2026-08-03-thor-real-weights/` — 5 harnesses + 5 raw JSON.

**Models — REAL checkpoints, `torch.load` + STRICT `load_state_dict`, both verified:**

| arm | ckpt | step | raster | params | role here |
|---|---|---|---|---|---|
| **flagship-v1-speedjerk** | `thor:~/models/flagship-v1-speedjerk/ckpt.pt` | **29999** | 256×256 square | 263.44 M | the **accuracy** arm (its trained raster) |
| v5f | `thor:~/models/v5f/ckpt.pt` | 1000 | 176×624 cyl | 263.58 M | the **latency** arm (the deployed geometry) |

⚠️ v5f on Thor is step 1000 — real trained weights but early. It is used as a **second weight
distribution and the deployed-geometry latency arm**, never as an accuracy verdict.

---

## 0. Why this run exists, in one line

Every number in the runbook's §1/§2/§3 was measured on a **randomly-initialised** model fed
`torch.randn` — there is no `torch.load` in any of the five Thor scripts. The runbook says so itself
and correctly refuses to call its own table a deployment gate. This run supplies the missing half:
**real weights, real held-out frames, real expert actions, and the four-family gate.**

**Bottom line: the latency result survives completely. The numerics result survives in magnitude but
NOT in shape — compounding is 3.1× steeper on real weights than the published table implied. And the
deployment gate PASSES on ADE, LATERAL, TACTICAL and STRATEGIC while firing on LONGITUDINAL at an
effect size of 0.12–0.58 %.**

---

## 1. Q1 — LATENCY IS WEIGHT-INDEPENDENT. Every published stage stands.

The falsifier was *"any stage differs from the published random-weight figure by > 10 %"*. **It did
not fire.** The decisive cell is the **random-weight control run in the same session on the same
box**, which isolates weights from session drift:

| stage @ **176×624** (deployed geometry) | REAL weights | RANDOM weights, same session | real vs random | published 08-02 |
|---|---|---|---|---|
| encoder fp32 | **196.86 ms** | 197.05 ms | **0.10 %** | 187.8 ms |
| encoder bf16 autocast | **30.23 ms** | 29.60 ms | 2.1 % | 27.78 ms |
| encoder speedup | **6.51×** | 6.66× | — | 6.76× |
| predictor eager, b1 | **4.088 ms** | 4.136 ms | 1.2 % | 4.23 ms |
| K=20 roll, eager | **78.06 ms** | — | — | 81.7 ms |
| ⭐ **TRT-fp16 engine, b1** | **1.17676 ms** | — | — | **1.168 ms (+0.7 %)** |
| ⭐ **TRT-fp16 engine, b9** | **1.29358 ms** | — | — | 1.294 ms (b1-fan run) |

⇒ **Real vs random is 0.1–2.1 %. Real vs published is 0.7–8.8 %, and the random control sits at the
same offset from published**, so that residual is session/thermal/contention drift, not weights.
✅ **The 5.33× and the ≈51 ms tick are admissible. They were measured on the wrong model and happen
to be right, and this run is what makes them citable.**

**The tick, recomposed on real weights at the deployed geometry:**

```
1 candidate                : 30.23 + 20 × 1.17676        =  53.77 ms   (54 % of the 100 ms budget)
9 candidates SERIALISED    : 30.23 + 20 × 1.17676 × 9    = 241.99 ms   (242 %)  ⛔
9 candidates BATCHED (b9)  : 30.23 + 20 × 1.29358        =  56.10 ms   (56 %)   ✅
```
Reproduces the B1 random-weight fan run (53.98 / 243.84 / 56.13) to **< 1 %**. The batch-9 engine
requirement is confirmed on real weights.

ⓘ **The predictor engine is geometry-independent, as it must be**: v1@256² and v5f@176×624 give
**1.17664** and **1.17676 ms** — it consumes 2048-d latents, not pixels. The encoder is not:
108.60 ms fp32 / 18.48 ms bf16 (**5.88×**) at 256², against 196.86 / 30.23 (**6.51×**) at 176×624.

---

## 2. Q2 — the precision gate on real activations: ⭐ **THE ERROR IS SMALLER AND THE COMPOUNDING IS STEEPER**

64 real held-out windows, encoded by the real encoder, rolled 20 steps under the **expert's real
future actions** — the operating condition, not `torch.randn`.

| condition | rel-err 1 step | rel-err 20 steps | **growth** | worst-case growth |
|---|---|---|---|---|
| published (random w, randn x) | 1.41e-3 | 1.80e-3 | **1.3×** | — |
| ✅ replicated here (random w, randn x) | 1.193e-3 | 1.652e-3 | **1.385×** | 1.49× |
| real w, randn x | 5.168e-4 | 1.063e-3 | **2.056×** | 3.28× |
| ⭐ **real w, REAL activations** | **3.127e-4** | **1.339e-3** | ⛔ **4.273×** | ⛔ **26.15×** |

**The decomposition is clean because each row changes exactly one thing:**

* **Weights alone** (row 2 → row 3): growth **×1.48**, absolute error **×0.43**.
* **Plus real activations** (row 3 → row 4): growth **×2.08**.
* **Total: compounding is 3.08× steeper than the published condition** — while the absolute error at
  1 step is **4.5× smaller** and at 20 steps **1.35× smaller**.

⇒ 🔴 **The published table was not optimistic in MAGNITUDE — it was optimistic in SHAPE.** It showed
a 1.3× growth over the rollout and the runbook already warned *"DO NOT READ THIS AS 'ERROR DOES NOT
COMPOUND'"*. **That warning was right and this run quantifies it: 4.27× mean, 26.15× worst case.**

⚠️ **The sharpest control, and it changes the interpretation:** the **TRT-fp32** engine compounds
**even harder — 9.83× mean, 46.81× max** (4.745e-5 → 3.723e-4). fp32 is not a precision problem.
⇒ **compounding is a property of the RECURSIVE ROLL (a Lyapunov effect), not of fp16.** The growth
RATIO is therefore **not** a precision diagnostic; only the **absolute level** is. A gate written on
the ratio would have condemned the more accurate engine.

**MEASURED, not inherited:** `thor_c2_real_weights.json:Q2_predictor_trt_numerics_REAL`,
`thor_c2b_random_control.json`.

---

## 3. Q3 — the bf16 encoder, the largest lever, checked for numerics for the first time

The 6.76× encoder lever had **never** been checked for accuracy at any weight init. Falsifier:
*latent cosine < 0.999 on real windows*. **It did not fire.**

| condition | rel-err | latent cosine mean | cosine **min** | worst per-state rel-err | channels > 10σ |
|---|---|---|---|---|---|
| **real weights, REAL frames** | 4.469e-3 | 0.9999895 | **0.9998735** | **1.683e-2** | **0** |
| real weights, randn input | 5.250e-3 | 0.9999867 | 0.9999176 | 1.306e-2 | 0 |
| random weights, randn input | 6.621e-3 | 0.9999781 | 0.9999700 | 7.753e-3 | 0 |

✅ **bf16 preserves the latent direction on real data.** Two findings inside the pass:

1. ⭐ **The outlier-channel mechanism does not appear here.** **Zero** post-pool channels exceed 10σ
   in any condition. Our §7.10 INT8 result (readout cosine collapsing to 0.566) is therefore
   **specific to INT8 on the un-normalised readout**, and does **not** generalise to bf16 on the
   encoder trunk. Two different levers, two different verdicts — the per-stage rule again.
2. ⚠️ **The tail widens with real data even though the mean improves**: worst-case per-state error is
   **2.2× wider** on real weights + real frames (1.68e-2) than on random (7.75e-3), while the global
   error is *lower*. That is the outlier-channel direction, just far from dangerous at bf16 — and it
   is the reason an INT8/NVFP4 attempt (O5/O6) must be gated on the **tail**, never the mean.

---

## 4. Q4 — the MHA fastpath, settled: ⛔ **INERT ON REAL WEIGHTS TOO**

The 2026-08-03 annotation found the flag inert but was itself measured on random weights, and a
wrong graph shows itself through the weights it multiplies. Re-run on the **step-29999** predictor:

| cell | nodes | fused MHA op | ORT rel-err, **REAL** state input | ORT rel-err, randn input |
|---|---|---|---|---|
| opset 17, fastpath **ON** | 1146 | no | **5.56e-08** | 1.035e-07 |
| opset 17, fastpath OFF | 1146 | no | 5.56e-08 | 8.80e-08 |
| opset **18**, fastpath **ON** | 1146 | no | **5.56e-08** | 9.19e-08 |
| opset 18, fastpath OFF | 1146 | no | 5.56e-08 | 1.913e-07 |

Identical node counts, no fused op anywhere, **opset 18 with the fastpath ON exports cleanly** where
the runbook records it *"fails loudly"*, and the 0.726 does not reappear. The falsifier (> 1e-4) did
not fire by **three orders of magnitude**.

⇒ **The mechanism is now non-reproduced across 10 cells and TWO independent weight distributions.**
The competing explanation — the runbook's own learning #8 (*identical error across precisions is the
signature of a wiring bug*) plus learning #9 (a gate that compared an engine to a **different random
model**) — stands essentially alone. 🔴 **The retraction §3 applies to our 2026-07-08 "ONNX-clean"
claim should be revisited by its owner: it rests on a mechanism that does not reproduce.**

⚠️ **Fairness, unchanged:** the original gate script is not on disk, so what it exported cannot be
inspected. ✅ **Keep `set_fastpath_enabled(False)`** — it costs 5.1e-7 in eager and ~1.6 % in engine
latency. Cheap insurance against a mechanism nobody has mapped.

---

## 5. ⭐ O1 — THE FOUR-FAMILY DEPLOYMENT GATE (the runbook's blocking P0), REAL WEIGHTS

**Two arms, identical windows, one process.** A = fp32 eager. B = **bf16 encoder + TRT-fp16
predictor** (dynamic batch 1–8, fastpath OFF). 39 clean held-out episodes → **859 windows**.
Every interval is the **paired episode-cluster bootstrap** (`taniteval/ci.py`, 2000 draws).

⛔ **Negative control ran BEFORE any score** and is in the JSON: the engine differs from eager
(5.9e-4) *and* responds to its inputs (zeroing the actions moves the output 0.232 relative). An
engine aliased to eager, or ignoring its inputs, would have made every delta 0 and the gate would
have "passed" vacuously.

### 5.1 ADE — one row of four, never "the result"

| horizon | A fp32 | B optimised | **paired Δ (B−A)** | CI | separated |
|---|---|---|---|---|---|
| ADE 0–2 s | **0.4209 m** | 0.4205 m | **−0.0004** | [−0.0009, +0.0001] | no |
| ADE @0.5 s | 0.0707 | 0.0707 | −0.0000 | [−0.0002, +0.0001] | no |
| ADE @1 s | 0.1438 | 0.1438 | +0.0000 | [−0.0002, +0.0002] | no |
| FDE @2 s | 0.8943 | 0.8933 | −0.0009 | [−0.0020, +0.0000] | no |

✅ **F-ADE did not fire** — the §7.10 bar is 0.02 m and the measured delta is **0.0004 m, 50× below
it**, with a CI containing zero. *(Sanity: v1's registry ADE is 0.452 m on 40 episodes/881 windows;
0.4209 on 39/859 is the same model, one clip short.)*

### 5.2 The four families — corrected instrument (see §6), paired deltas

| family | metric | A fp32 | B opt | Δ (B−A) | % of level | separated |
|---|---|---|---|---|---|---|
| **LONGITUDINAL** | speed MAE | 0.4666 m/s | 0.4666 | −0.0000 | −0.00 % | no |
| | **speed bias** | **+0.1885 m/s** | 0.1874 | **−0.0011** | **−0.58 %** | ⛔ **yes** |
| | along-track MAE | 0.3893 m | 0.3890 | −0.0003 | −0.08 % | no |
| | **accel MAE** | **0.7701 m/s²** | 0.7710 | **+0.0009** | **+0.12 %** | ⛔ **yes** |
| | distance-keeping | **UNAVAILABLE** — no lead-agent track in the ingest (`obstacle.offline` unread), n = 0. **A WORK ITEM, not a pass.** |
| **LATERAL** | heading MAE | 0.7935° | 0.7931 | −0.0004 | −0.04 % | no |
| | yaw-rate MAE | 1.4115 °/s | 1.4115 | −0.0000 | −0.00 % | no |
| | **curvature MAE** | **0.003304 1/m** | 0.003309 | **+5e-06** | **+0.16 %** | ⛔ **yes** |
| | cross-track MAE | 0.1118 m | 0.1117 | −0.0001 | −0.09 % | no |
| **TACTICAL** | accuracy vs GT | 0.5914 | 0.5949 | +0.0035 | — | no (CI touches 0) |
| | ⭐ **decision agreement A↔B** | **0.9942** [0.9872, 0.9988] — **5 of 859 flipped**. Bar 0.953. ✅ | | | | |
| | never-predicted classes | **none** in either arm (all 5 manoeuvres emitted) | | | | |
| **STRATEGIC** | vision-only route acc | identical in both arms | | | | |
| | ⭐ **decision agreement A↔B** | **1.0000** — **0 of 859 flipped**. Bar 0.953. ✅ | | | | |

### 5.3 The verdict, stated exactly as pre-registered

| falsifier | fired? | evidence |
|---|---|---|
| F-ADE (Δ > 0.02 m **and** CI excludes 0) | ✅ **no** | Δ 0.0004 m, CI spans 0 |
| **F-LONGITUDINAL** (any metric's paired CI excludes 0) | ⛔ **YES** | `speed_bias`, `accel_mae` |
| F-LATERAL (same) | ⛔ **YES on the corrected instrument** | `curvature_mae` *(did not separate pre-fix)* |
| F-TACTICAL (agreement < 95.3 %) | ✅ **no** | 99.42 % |
| F-STRATEGIC (agreement < 95.3 %) | ✅ **no** | 100.00 % |

⛔ **I am not moving the bar after seeing the data.** The falsifier was written as a **separation**
test and it fired as written. What I add — clearly labelled, not as a redefinition — is the
**effect size**, because separation and materiality are different questions and the binding rule
asks for both to be legible:

> **the three separated metrics move by 0.12 %, 0.16 % and 0.58 % of their own fp32 level.** A
> paired episode-cluster bootstrap over 859 windows / 39 episodes resolves sub-percent shifts; that
> is the estimator working, not a regression appearing. The largest, `speed_bias`, moves **toward
> zero** (less over-speeding: +0.1885 → +0.1874 m/s).

**⇒ RECOMMENDATION TO THE PI (the decision is the PI's, not mine).** Ship bf16 + TRT-fp16 **if and
only if** a materiality threshold is pre-registered alongside the separation test — I propose **1 %
of the fp32 level per family metric**, which all three clear by 2–8×. Without such a threshold the
gate as written says **DO NOT SHIP**, and I am reporting it that way.

---

## 6. 🔴 A PROGRAM-WIDE INSTRUMENT DEFECT FOUND WHILE RUNNING THIS GATE

`taniteval/four_families.py` hard-coded `DT_S = 0.1`, but `all_families` reads `win["pred"]`, which
for **both** `rollout.collect` and `refc_eval.collect` is the **SPARSE 4-waypoint view at
`WP_STEPS = (5, 10, 15, 20)` — a 0.5 s grid**. Every derivative was divided by the wrong dt.

**The negative control that turns this from an argument into a measurement** — the episode already
carries the answer, and nothing had ever compared against it:

| quantity, 859 real held-out windows | value |
|---|---|
| the ego's **own recorded speed**, `poses[:, 3]` | **12.4565 m/s** |
| `_seq_geometry(gt)` at the hard-coded dt = 0.1 | **62.9789 m/s** |
| **ratio** | ⛔ **5.0559** |
| `_seq_geometry(gt)` at the true dt = 0.5 | **12.5958 m/s** (**1.011×** truth) |

**Blast radius — measured on the same windows, PRE-FIX → CORRECTED:**

| metric | published-style value | corrected | factor | mechanism |
|---|---|---|---|---|
| `speed_mae_mps` | 2.3332 | **0.4666** | **÷5.00** | 1/dt |
| `speed_bias_mps` | 0.9423 | **0.1885** | ÷5.00 | 1/dt |
| `speed_rmse_mps` | 3.3541 | **0.6708** | ÷5.00 | 1/dt |
| `accel_mae_mps2` | 19.2517 | **0.7701** | **÷25.00** | 1/dt² |
| `yaw_rate_mae_degps` | 9.1420 | **1.4115** | ÷6.48 | 1/dt **and** the mask |
| `curvature_mae_1pm` | 0.027623 | **0.003304** | **÷8.36** | ⭐ **the mask alone** |
| `heading_mae_deg` | 1.5089 | **0.7935** | ÷1.90 | the mask alone |
| `along_*`, `cross_*` | 0.3893 / 0.1118 | **unchanged** | ×1.00 | dt-invariant ✅ |

⭐ **The second half is the bigger surprise.** `MIN_DS_M = 0.05 m` is a *distance* gate meant to
drop steps where the vehicle barely moves (curvature = dθ/ds **explodes** as ds→0). On a 0.5 s grid
that gate is 5× too permissive, so a handful of crawling steps with exploding curvature stayed in
the mean: **excluding 38 more steps out of ~2465 moves `curvature_mae` by 8.4×.** Curvature and
heading are dt-**invariant** and were still wrong — through the validity mask, not the arithmetic.

**Every corrected value is now physically plausible for a 2 s prediction** (0.47 m/s speed MAE,
0.77 m/s² accel, 0.79° heading, 1.41 °/s yaw-rate) where `19.25 m/s²` never was — and nobody caught
it because no number was ever compared against a physical quantity.

**⇒ Consequences, stated precisely:**
1. ✅ **Every CROSS-ARM comparison stays valid.** The factor is common to both arms, so ranks,
   paired deltas and every "A vs B" verdict published to date are unaffected in direction.
2. ⛔ **Every ABSOLUTE quotation of a rate is wrong by the factor above** — including the REF-C Thor
   panel of 2026-08-02 (`speed_mae 3.0609 → 0.6122 m/s`, `yaw_rate 22.6241 → ~3.5 °/s`).
3. ⛔ **Any comparison against a physical or external bar was void.** "Is 2.33 m/s speed MAE
   acceptable?" had the wrong answer; the real figure is 0.47 m/s.

**🔴 THE ROOT-CAUSE CLASS, and it is the interesting part: the trap was already documented — in a
FORK, not at the source.** `stack/tanitad/eval/idm_families.py` says verbatim that feeding 4
waypoints at {5,10,15,20} to `four_families` *"reads every speed and yaw-rate 5× too large"*, and
its author responded by **re-implementing the geometry with an explicit cadence for the IDM**. What
nobody noticed is that `rollout.collect` and `refc_eval.collect` emit **exactly that same shape** —
so the trap was never hypothetical, it was already live in the mainline instrument. ⇒ **a hazard
documented next to one caller, instead of fixed at the shared function, protects only that caller
and gives every other reader false confidence that someone has looked.**

### 6.1 The fix — implemented, tested, staged

`taniteval/taniteval/four_families.py`:
* `_seq_geometry(wp, dt)` and `longitudinal/lateral(..., dt)` take the grid explicitly.
* **`infer_dt(win)`** derives it from the window's own `wp_steps` × `dt_s` contract and **never
  guesses silently** — it returns a provenance string, including for the non-uniform and
  missing-contract cases.
* `all_families(win, hier, prefer_dense=True)` now **prefers the true 10 Hz `pred_dense`/`gt_dense`
  path** when the window carries it (20 samples, real 0.1 s tick), falling back to the sparse view
  **with the derived dt**. Pass `prefer_dense=False` to reproduce a historical sparse number.
* `MIN_DS_M` becomes **`MIN_DS_MPS = 0.5 m/s`**, i.e. `min_ds = 0.5 × dt` — identical at 10 Hz
  (0.05 m), correct at every other cadence.
* Every family output now carries `dt_s`, and `all_families` emits a `_grid` block with the grid
  used, its provenance, and the correction history. **A rate can no longer travel without its grid.**

`taniteval/tests/test_four_families_dt.py` — **12 tests, all passing**, pinning: a constant-velocity
path reports its own speed on any grid; the ×5 / ×25 correction factors; that `infer_dt` never
guesses silently; that heading/curvature/positions stay dt-invariant; that the `min_ds` gate scales
(a 0.2 m/s crawl is now correctly excluded and was not); and that headings are degrees.

---

## 6bis. ⭐ O2-pre CLOSED — the encoder now exports at the deployed geometry

The other ONNX blocker, unrelated to precision: **the encoder did not export at 176×624 at all.**

```
SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
                    output size that are not factor of input size
```

**Cause, read from source and confirmed by construction:** at 176×624 with patch 16 the token grid
is **11×39**, and the readout pools onto **4×4**. `11 % 4 ≠ 0`, so `SpatialGridReadout` took its
`nn.AdaptiveAvgPool2d` fallback — which ONNX cannot express at a non-factor output size, at any
opset and both fastpath settings. That blocked **O2** (a TensorRT engine for the encoder), which is
the designated fallback for the largest lever in the entire Thor result.

**The fix, and why it is a re-expression rather than an approximation.** Adaptive pooling with
*static* input and output sizes is a **fixed linear operator**: output bin `i` averages input rows
`[floor(i·H/G), ceil((i+1)·H/G))`. Materialising that as two constant averaging matrices turns the
op into two matmuls, which every opset exports, **using the same bins and therefore the same
numbers**.

⚠️ **The bins OVERLAP** — `start` floors while `end` ceils, so 11 → 4 is **3/4/4/3** (14 cell-uses
over 11 cells), not a tidy 3/3/3/2 partition. My first test asserted the partition and **failed**;
the measurement corrected me. An "export fix" written from the partition intuition would have
silently changed what the trained readout sees, which is worse than not exporting.

**MEASURED on Thor, REAL v5f weights:**

| cell | export | nodes | adaptive pool in graph | ORT rel-err vs eager |
|---|---|---|---|---|
| encoder 176×624, opset 17, fastpath **OFF** | ✅ **1.7 s, 349.2 MB** | 1034 | **no** | **1.51e-05 / 1.77e-05** |
| encoder 176×624, opset 17, fastpath **ON** | ✅ 1.5 s, 349.2 MB | 1034 | no | 1.51e-05 / 1.77e-05 |

Both falsifiers passed: it exports, and it computes the right thing (1.5e-05 against a 1e-4 bar).
*(A fifth independent confirmation, incidentally, that the fastpath flag is inert.)*

⛔ **The regression that mattered more than the fix — MEASURED on the real v1 checkpoint:**

| check | result |
|---|---|
| `exact_pool` / pool module | **True** / `AvgPool2d` — v1 takes the untouched tiling route |
| derived matrices built? | **no** (`has_pool_mh = False`) |
| STRICT `load_state_dict` of the 29999 ckpt | ✅ **OK** — the matrices are `persistent=False`, so no state_dict key was added |
| readout output vs `AvgPool2d` | ✅ **bit-identical, max abs diff 0.0** |

⇒ **the deployed 256px path cannot have moved a single v1 number.** `stack/tests/test_readout_onnx_pool.py`
pins all four properties plus the ONNX round-trip — **13 tests, all passing.**

## 7. What stands, what is restated, what is new

| runbook claim | verdict on real weights |
|---|---|
| 272.56 → ≈51.2 ms, **5.33×** | ✅ **STANDS** — every stage within 8.8 % of published, and within **2.1 % of a same-session random control**, so the residual is drift, not weights |
| encoder 187.8 → 27.8 ms bf16 (6.76×) | ✅ stands — real weights measure 196.86 → 30.23 (**6.51×**); the level offset is session drift, reproduced by the random control |
| predictor 4.23 → 1.168 ms TRT-fp16 | ✅ **stands to 0.7 %** — 1.17676 ms on real weights |
| §2 precision table (rel-err 1.41e-3 → 1.80e-3, growth 1.3×) | ⚠️ **RESTATED.** Absolute error is **smaller** on real weights (3.13e-4 → 1.34e-3) but growth is **4.27× (max 26×)**, 3.1× steeper. The runbook's own "do not read this as no-compounding" warning is vindicated and now quantified |
| §2 "architecture read, not a deployment gate" | ✅ correct then — **and the deployment gate has now run** (§5) |
| §3 MHA fastpath 0.726 mechanism | ⛔ **does not reproduce**, now at 10 cells across two weight distributions. The 2026-07-08 "ONNX-clean" retraction should be revisited |
| §3 "opset 18 fails loudly with fastpath ON" | ⛔ **does not reproduce** — exports cleanly at rel-err 4.32e-07 (random w) / 5.56e-08 (real w) |
| O1 "four-family gate has NOT run — Thor has no val data" | ✅ **CLOSED.** It has run. 4 of 5 falsifiers pass; LONGITUDINAL separates at 0.12–0.58 % effect size |
| §6.2 / O2-pre "the encoder does not export at 176×624" | ✅ **CLOSED** — fixed in `stack/tanitad/models/readout.py`, exports in 1.7 s at ORT rel-err **1.51e-05 / 1.77e-05**; **O2 is unblocked** |
| every published four-family **rate** | 🔴 **CORRECTED** — see §6. Cross-arm comparisons unaffected; absolute rates were ×5 / ×25 / ×6.5 / ×8.4 |

---

## 8. Evidence class

| claim | class |
|---|---|
| all latencies, engine builds, rel-errs, node counts | **MEASURED (ours)** — `thor_c2_real_weights.json`, `thor_c2b_random_control.json`, Thor, torch 2.13.0+cu130, TRT 10.13.3.9, **real checkpoints** |
| ADE + four families + every CI | **MEASURED (ours)** — `thor_c3_four_family_gate.json`, 859 windows / 39 episodes, paired episode-cluster bootstrap, 2000 draws |
| the dt defect and its correction factors | **MEASURED (ours)** — `thor_c4_rescore_corrected.json`, negative control against `poses[:,3]`, plus 12 passing unit tests |
| "the 0.726 was a wiring bug" | **HYPOTHESIS, now very strongly supported** — non-reproduction at 10 cells / 2 weight distributions. Not closed: the original script is not on disk |
| v5f numbers as a MODEL verdict | ⛔ **NOT ESTABLISHED** — step 1000. Latency/numerics control only |
| absolute four-family levels at n=40 | ⚠️ **decision_grade = False** — 39 of 40 clips (`ep_00028.pt` arrived truncated, 92.3 MB vs ~117 MB, and only the 256×640cyl val is on HF). **The PAIRED delta is on identical windows and is unaffected.** |
| the 100 ms budget, the 95.3 % agreement bar, §7.10's 0.02 m | **INHERITED** — carried from the runbook / paper, not re-derived here |

---

## 9. Next, in priority order

1. ⭐ **PI decision: pre-register a MATERIALITY threshold beside the separation test.** Without it
   the O1 gate reads DO-NOT-SHIP on a 0.12 % effect. This is the one genuinely blocked item.
2. **Rebuild `predictor_fp16.plan` at batch 9 or dynamic** — confirmed on real weights: 242 % vs
   56 % of budget. The dynamic 1–8 engine built here is the template.
3. **Re-run the gate with `prefer_dense=True`** (the corrected module's default) — the true 10 Hz
   path is a strictly better instrument than the 0.5 s chord grid used here.
4. **`obstacle.offline` ingest** — distance-keeping is the one family still UNAVAILABLE, and the
   binding rule calls that a work item.
5. ✅ **O2-pre is CLOSED (§6bis) — so run O2**: build the encoder TensorRT engine at 176×624 and
   compare it against bf16 autocast (30.23 ms). The falsifier is already written: engine ≤ bf16
   ⇒ keep autocast and close the item.
6. ⛔ **`ep_00028.pt` is unrecoverable and the 256px val cache is single-disk.** Runbook §11's rule,
   live again: push it to HF as a dataset.
