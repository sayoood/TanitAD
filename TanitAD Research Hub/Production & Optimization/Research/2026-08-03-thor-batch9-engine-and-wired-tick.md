# Thor P6 — the batch-9 engine is BUILT, and the first END-TO-END measured tick

**Date:** 2026-08-03 (Europe/Berlin). **Hardware:** NVIDIA Thor (Blackwell **sm_110**), aarch64,
torch **2.13.0+cu130**, TensorRT **10.13.3.9**, `~/venvs/tanitad-edge`. **Cost $0** (owned hardware).
**Plan ref:** `THOR_DEPLOYMENT_RUNBOOK.md` §6 pricing annotation item 1 (the 4.3x batch-9 finding)
and backlog **O9**.
**Artifacts:** `Implementation/incoming/2026-08-03-thor-batch9-engine/` — 5 harnesses + raw JSON
(`INTAKE.md` lists every one); engines at `thor:~/trt_deploy/` (with `MANIFEST.md`).
**Code:** `stack/scripts/build_predictor_trt.py` (new) · `TacticalSelector.propose_and_score(...,
batch_fan=True)` + `_fan_batched` in `stack/tanitad/models/fourbrain.py` · 3 new tests.
**Model:** `flagship-v1-speedjerk` @ **step 29999**, `torch.load` + **STRICT** `load_state_dict`,
**256x256 SQUARE** — its trained raster, asserted before any frame is fed. Second arm **v5f** @ step
1000 at **176x624 cylindrical**. ⛔ The two rasters are never crossed.

---

## 1. THE HEADLINE — the fix is real, it is bigger than 4.3x, and it is not only the engine

| # | claim | status |
|---|---|---|
| 1 | The batch-9 (dynamic **1..9**) engine is **built, verified by loading and executing, and banked** | ⭐ **DONE** |
| 2 | **The engine alone would NOT have delivered it.** `TacticalSelector` loops over candidates, so a batch-1-shaped CALLER serialises whatever engine it is handed. The batching had to be implemented in `stack/` | ⭐⭐ **NEW — and it is the load-bearing half** |
| 3 | First **end-to-end MEASURED** tick (encoder + heads + 9-candidate fan + decode + scoring), p50 **and** p95 | ⭐ **60.3 / 63.1 ms = 60 % / 63 % of the 100 ms budget** |
| 4 | The engine shipped on 2026-08-02 is batch-1 **and built from RANDOM WEIGHTS** | 🔴 **NEW — it could never have been deployed at all** |
| 5 | Batching does not change the engine's answer (b9 vs b1 **1.57e-4** on real activations) | ✅ the 2026-08-03 four-family gate transfers |
| 6 | A "48 % tactical agreement / fp16 fails" result | 🔴 **RETRACTED BEFORE PUBLICATION — it was my own dropped-intent wiring bug (§5). With a correct engine: 200/200 at K=4, 0.0 regret** |
| 7 | The **intent token** is quantitatively load-bearing | ⭐ withholding it moves the operative prediction **5.2 %** and costs **3.81 m** mean tactical regret at K=20 |

---

## 2. D1 — the engine

`stack/scripts/build_predictor_trt.py` (checked in) exports with
`torch.backends.mha.set_fastpath_enabled(False)`, builds via `trtexec`, and then **verifies by
deserialising the plan, reading its optimisation profile back, and executing it** — never by exit
code.

| engine | profile (`states` min/opt/max) | build | rel-err vs eager @b1 | @b9 |
|---|---|---|---|---|
| ⭐ **`v1_dyn1-9_fp16`** (deployment) | **(1,8,2048) / (9,8,2048) / (9,8,2048)** | 38.4 s, 174.3 MB | **3.67e-4** | **6.21e-4** |
| `v1_static_b1_fp16` (the shipped shape, control) | (1,8,2048) all three | 36.0 s, 174.0 MB | 5.18e-4 | ⛔ cannot run |
| `v5f_dyn1-9_fp16` (deployed geometry) | (1,8,2048) / (9,8,2048) / (9,8,2048) | 38.2 s, 174.3 MB | 1.60e-3 | 1.33e-3 |

**Negative control on the build itself:** the v1 and v5f engines fed the *same* input differ by
**0.446** relative — engines carry weights, so an identical answer would have meant one was built
from the wrong checkpoint.

### 2.1 Per-batch latency — and why two numbers, not one

| batch | engine **in-process** p50 / p95 (ms) | per candidate | eager p50 | per candidate |
|---|---|---|---|---|
| 1 | **1.632** / 1.917 | 1.632 | 4.118 | 4.118 |
| 4 | 1.433 / 1.669 | 0.358 | 6.380 | 1.595 |
| 8 | 1.511 / 1.747 | 0.189 | 10.902 | 1.363 |
| ⭐ **9** | **1.494** / 1.790 | ⭐ **0.166** | 10.970 | 1.219 |

⭐ **Nine candidates cost 0.92x of one** in-process — the fan is not merely amortised, it is free,
because the per-call binding overhead dominates and is paid once instead of nine times. Per
candidate the engine is **9.83x cheaper at batch 9 than at batch 1**.

⚠️ **The published 1.168 / 1.294 ms are `trtexec` medians (kernel-only). Deployment does not pay
that.** Measured here on the same plans: `trtexec` gives **1.244 (b1) / 1.289 (b9)** for the dynamic
engine and **1.172** for the static b1 — reproducing the published figures to **1.2 %** — while the
same engines cost **1.632 / 1.494 ms** called from python. ⇒ **any tick composed from `trtexec`
medians understates the deployed tick by ~0.2 ms per predictor call**, which is 4 ms over a K=20
roll. The two metrics are both correct and must not be mixed.

ⓘ **The cost of dynamism, quantified:** the dynamic engine at batch 1 is **6.2 %** slower than a
static batch-1 engine (1.244 vs 1.172 ms, trtexec). That is the price of serving 1..9 from one
plan, and it is worth paying — a static b9 engine cannot serve b1 at all.

### 2.2 The fan, MEASURED (9 candidates x K=20 predictor steps, no decode)

| how the fan runs | p50 | p95 |
|---|---|---|
| eager, serialised | 717.0 ms | 722.7 |
| eager, batched | 206.3 ms | 207.5 |
| ⛔ **TRT batch-1, serialised** (the shipped shape) | **265.7 ms** | 269.7 |
| TRT dynamic, serialised (engine fixed, caller not) | 272.8 ms | 277.3 |
| ⭐ **TRT dynamic, BATCHED** | **31.4 ms** | 36.9 |
| (one candidate, same path) | 30.0 ms | 31.4 |

⇒ **8.47x on the fan**, and — the point of row 4 — **rebuilding the engine while leaving the caller
serialised buys nothing (272.8 ms, slightly WORSE than the batch-1 engine).** The 4.3x in the
runbook's annotation is a property of the engine *and* the caller together.

---

## 3. D2 — the FULL wired tick, end to end, on real frames

Every previously published Thor tick was **composed arithmetically** from `encoder + K x predictor`.
This wires the real path — `encode_window -> strategic_policy -> tactical_policy ->
TacticalSelector.propose_and_score` over the 9-candidate vocabulary, including `step_readout` decode,
SE(2) accumulation and scoring — on **60 real held-out windows**, K=20.

**D6 — the headline, on the CORRECT (intent-carrying) engine** (`thor_d6_tick_intent_K20.json`):

| arm | p50 | **p95** | p99 | vs 100 ms budget |
|---|---|---|---|---|
| A — fp32 eager, serialised fan (today's code + today's precision) | 764.1 | 768.4 | 769.8 | ⛔ **764 %** |
| B — bf16 + engine, **serialised** (engine rebuilt, CALLER not fixed) | 372.0 | 380.1 | 382.5 | ⛔ **372 %** |
| ⭐ **C — bf16 + engine, BATCHED fan** | **60.3** | **63.1** | 65.3 | ✅ **60 % / 63 %** |
| D — bf16 + eager batched fan (**no engine**) | 204.4 | 205.7 | 206.3 | ⛔ 204 % |

⭐ **6.17x on the measured tick (B -> C), and it clears the budget at p95, not just p50.**

**D2 — the first pass, on the intent-less engine** (`thor_d2_full_tick_K20.json`), kept because it
carries the *shipped* batch-1 engine as its arm B and because it is the run §5's wiring bug was found
in:

| arm | p50 | p95 | vs budget |
|---|---|---|---|
| A — fp32 eager, serialised | 860.5 | 866.0 | ⛔ 861 % |
| B — bf16 + **shipped batch-1 engine**, serialised | 365.9 | 375.8 | ⛔ **366 %** |
| C — bf16 + dynamic 1..9 engine, batched | 62.1 | 67.4 | ✅ 62 % |
| D — bf16 + eager batched (**no engine**) | 233.5 | 239.6 | ⛔ 234 % |

⛔ **Row D is the fallback that is not a fallback:** without the engine, even a batched fan is
**2.3x over budget**. The engine is load-bearing.
⚠️ **Run-to-run drift is ~13 % on this box** (arm A: 860.5 vs 764.1 ms for the same computation on
the same 60 windows, minutes apart). ⇒ **quote the WITHIN-RUN ratio, and treat the absolute tick as
±13 %.** 60.3 ms at +13 % is still 68 ms, inside budget; the conclusion survives the drift, which is
the check that matters.

### 3.1 Stage breakdown (arm C, forced per-stage synchronisation)

| stage | p50 | p95 |
|---|---|---|
| encoder, bf16 autocast | 17.1 | 18.8 |
| strategic + tactical heads | 7.2 | 7.7 |
| 9-candidate fan **incl. decode + scoring** | 45.1 | 47.4 |

⚠️ The three sum to 69.5 ms against a 62.1 ms tick: forcing `cuda.synchronize()` between stages
exposes latency that otherwise overlaps. Read the breakdown as **proportions**, the 62.1 as the tick.

⭐ **The decode and scoring are ~30 % of the fan** (45.1 ms measured vs 31.4 ms of pure predictor
rolls) — a cost **no previous composition contained**, because no previous composition ran the
selector.

ⓘ **Encoder 17.1 ms here vs 27.8 / 30.2 published**: this is v1 at **256x256** (its trained raster);
the published figures are v5f at **176x624**, which is 2.6x the pixels. Not a discrepancy — a
different arm. The 176x624 tick keeps the published ~30 ms encoder, so arm C at the deployed
geometry projects to **~75 ms**, still inside budget.

---

## 4. D3 — precision, on REAL trained weights and REAL activations

⛔ Latency is weight-independent; numerics are not. All of §4 is real weights + real encoder output.

### 4.1 Does batching change the engine's ANSWER? — the question that decides transfer

| comparison (32 real held-out windows, real encoder states) | rel-err |
|---|---|
| ⭐ **dynamic engine @b9 vs the SAME engine @b1** | **1.57e-4** |
| dynamic engine @b9 vs fp32 eager | 3.29e-4 |
| dynamic engine @b1 vs fp32 eager | 3.31e-4 |
| static b1 engine vs fp32 eager | 3.45e-4 |

⇒ **batching moves the answer by HALF of what fp16 itself moves it, and the b9 and b1 errors against
eager are identical to 3 significant figures.** The batch-9 engine is the same instrument as the
batch-1 one. ✅ **Therefore the 2026-08-03 four-family gate — which ran a dynamic 1..8 engine — 
transfers to the 9-candidate fan.** (Had b9 differed from b1, it would not have.)

### 4.2 Compounding over a 20-step roll

| | 1 step | 20 steps | growth |
|---|---|---|---|
| batched engine vs fp32 eager, real activations, synthetic constant primitive | **7.85e-4** | **9.50e-4** | **1.21x** |

⛔ **This does NOT refute the 2026-08-03 finding of 4.27x mean / 26.15x worst-case growth, and must
not be quoted as if it did.** Two things differ: that run rolled under **the expert's real future
actions** and reported a **per-window** growth ratio (mean and max over windows); this one rolls
under a **constant synthetic primitive** and reports a **pooled** norm ratio over a 9-row batch,
which averages away per-window blow-ups. What *does* replicate is the quantity the earlier run said
to gate on — **the absolute level**: 9.50e-4 here against 1.339e-3 there, a factor of 1.4 on a
different action sequence.

ⓘ The per-step curve is non-monotonic — it peaks at step 5 (1.12e-3) and then *declines* to 9.5e-4.
A saturating error is consistent with the roll settling onto an attractor rather than diverging;
under the expert's real actions it did not saturate. **Gate on the level, never the ratio** (that
rule was already established when TRT-**fp32** was measured compounding harder than fp16).

---

## 5. 🔴 A "PRECISION FAILURE" THAT WAS A WIRING BUG — caught in my own instrument

D2's first pass compared arm A (fp32 eager, serialised) against arm C (bf16 + engine, batched) and
measured **48.3 % selected-candidate agreement** on 60 windows against the 95.3 % bar, with a **max
score delta of 73.4**. A 73-unit delta cannot be a 3e-4 numerical effect, and **three things changed
at once**. Reporting it as "fp16 fails the tactical gate" would have been the easy, wrong answer.

### 5.1 Decompose first — one factor at a time (200 windows / 23 episodes, `thor_d4`)

| step | what changes | K=4 agreement | K=20 agreement |
|---|---|---|---|
| P0 -> P1 | **fan batching only** (my code change) | ✅ **1.0000** [1.0, 1.0], 0 flips | ✅ **1.0000** [1.0, 1.0], 0 flips |
| P1 -> P2 | fp32 eager -> **TRT-fp16 engine** | ⛔ 0.8650 [0.787, 0.930], 27 flips | ⛔ 0.4000 [0.266, 0.540], 120 flips |
| P2 -> P3 | fp32 encoder -> **bf16 encoder** | ✅ **1.0000**, 0 flips | ✅ **1.0000**, 0 flips |

⭐ **The batching is exactly decision-neutral** — max score delta **0.0** at K=4 and **1e-4** at K=20.
The falsifier written for the code change did not fire. **The bf16 encoder is decision-neutral too.**

### 5.2 …and the "engine" step was MY OWN wiring bug, not fp16

The engine was exported as `(states, actions) -> z_next`. The deployed operative predictor is
**FiLM-conditioned on the D-030 tactical intent token**, and `TacticalSelector` passes `intent=` as
a keyword — which a two-input wrapper accepts and **silently ignores**. The engine arm was therefore
computing the *unconditioned* prediction. Median |Δscore| **3.48** (K=4) and **41.7** (K=20) against
decision margins of 1.91 and 0.97 — a systematic shift, not noise.

**Rebuilt with `intent` as a third input** (`build_predictor_trt.py --intent-dim 256`;
`intent_is_live_rel_change = 0.0522`, i.e. the token moves the operative prediction by 5.2 %, so it
is verifiably live), and re-run on 200 windows (`thor_d5`):

200 windows / 23 episodes, bar 95.3 %. **Regret** = `score_fp32[fp16's pick] − score_fp32[fp32's
pick]` ≥ 0 — how much worse, in fp32's own metres, the tested arm's choice is:

| arm | K | agreement (episode-cluster bootstrap) | flips | regret |
|---|---|---|---|---|
| batching only — the code change | 4 | ✅ **1.0000** [1.0, 1.0] | 0 | **0.0** |
| batching only — the code change | 20 | ✅ **1.0000** [1.0, 1.0] | 0 | **0.0** |
| ⭐ **TRT-fp16 batch-9 engine, WITH intent** | **4** | ✅ **1.0000** [1.0, 1.0] | 0 | ⭐ **0.0 exactly** |
| ⭐ **TRT-fp16 batch-9 engine, WITH intent** | **20** | ✅ **0.9850** [0.960, 1.0] | 3 | ⭐ mean **4.1e-05 m**, p95 **0.0**, **max 0.0038 m** |
| the same engine **without** intent (the bug, reproduced deliberately) | 4 | ⛔ 0.8650 [0.787, 0.930] | 27 | mean **0.131**, p95 **1.07**, max **4.76** |
| the same engine **without** intent | 20 | ⛔ 0.4000 [0.266, 0.540] | 120 | mean **3.81**, p95 **18.6**, max **38.1**, **48.5 % of windows > 1 m worse** |

⇒ 🔴 **"fp16 flips 13.5 % of tactical decisions" is RETRACTED before it was ever published.** With a
correctly exported engine the fp16 batch-9 engine **passes the 95.3 % bar at both horizons** —
perfectly at K=4, and at K=20 its three flips are demonstrably ties, costing **3.8 mm** at worst.
The entire original effect was a dropped hierarchy input.

ⓘ **The intent token is doing real work**, incidentally: withholding it moves the operative
prediction by **5.2 %** relative and costs **3.81 m** of mean tactical regret at K=20. That is a
quantitative datum for the hierarchy thesis, obtained by accident from a bug.

**Root-cause class:** *an engine compared against a model it does not implement* — the runbook's own
learning #8/#9 (`identical-across-precisions error = a wiring bug`; `a gate compared the engine to a
different random model`), now recurring with a third mechanism. **The fix is in the runtime, not in
the report:** `TRTPredictor.forward` now **raises** when an `intent` token is passed to an engine
that has no `intent` input (and when an intent engine is called without one), and `verify_engine`
asserts the token is live. A wrapper that accepts an argument and ignores it is the failure mode;
silence was the only symptom.

⚠️ **Scope, honestly:** the regret is 0.0 on 200 windows at K=4 and this is a *decision-agreement*
result, not a four-family accuracy gate. The 9-primitive vocabulary is an instrument of mine (no
primitive table is committed anywhere in `stack/`), held identical across arms.

---

## 6. What this CONFIRMS, RESTATES and RETRACTS

| prior claim | verdict |
|---|---|
| "9 serialised = 243.84 ms (244 %), batched = 56.13 ms (56 %)" (runbook §6 annotation 1) | ✅ **CONFIRMED in direction and magnitude, RESTATED in level.** Measured end to end with the real selector: **365.9 ms** serialised and **62.1 ms** batched. Both are ~10-50 % higher than the composition because the composition used `trtexec` kernel medians and omitted decode, scoring and the heads |
| "O9 is worth 4.3x" | ⚠️ **RESTATED — it is 6.17x on the measured tick** (and 8.47x on the fan alone) |
| "rebuild the engine with a batch-9 profile" | ⚠️ **INSUFFICIENT AS WRITTEN.** The engine is half the fix; the serialised CALLER is the other half, and with it left alone the rebuild measures *slower* (272.8 vs 265.7 ms) |
| `predictor_fp16.plan` is the shipped engine | 🔴 **RETRACTED.** It is batch-1 **and random-weight** — `thor_trt.py` has no `torch.load`. It was never deployable. Superseded by `~/trt_deploy/` |
| published 1.168 / 1.294 ms engine latency | ✅ **CONFIRMED to 1.2 %** as `trtexec` medians — ⚠️ and they are **not** the deployed cost (1.632 / 1.494 ms in-process) |
| the 2026-08-03 four-family gate (dynamic 1..8 engine) | ✅ **TRANSFERS** to the 9-fan — b9 vs b1 is 1.57e-4, half the fp16 error |
| §2 "error does not compound" | ✅ still withdrawn; §4.2 here is a *different* statistic and does not reinstate it |
| `set_fastpath_enabled(False)` | ✅ **KEPT** in the checked-in export path (`build_predictor_trt.py`), as instructed |

---

## 7. Evidence class

| claim | class |
|---|---|
| every latency, engine build, profile, rel-err in §2-§4 | **MEASURED (ours)** — `thor_d1_batch9_engine.json`, `thor_d2_full_tick_K20.json`, **`thor_d6_tick_intent_K20.json`** (the headline tick), Thor, torch 2.13.0+cu130, TRT 10.13.3.9, **real step-29999 weights** |
| the decision decomposition and the regret in §5 | **MEASURED (ours)** — `thor_d4_decision_decomposition.json`, `thor_d5_selector_regret.json`, **200 windows / 23 episodes**, episode-cluster bootstrap (2000 draws) |
| the intent engine's build + token liveness | **MEASURED (ours)** — `predictor_v1_intent_dyn1-9_fp16.json`, `intent_is_live_rel_change` **0.0522** |
| "`TacticalSelector` has no production caller" | **MEASURED (ours), TWO probes** — repo-wide grep (`stack/tests` + these harnesses + stale worktrees only) and a separate `taniteval/` grep; the closed-loop driver `closedloop_drive.py::FlagshipV1Policy.plan` is heads-only |
| "the shipped plan is random-weight" | **MEASURED (ours), TWO probes** — (1) `thor_trt.py` contains no `torch.load`/`load_state_dict`; (2) the plan's profile is static (1,8,2048) and it is not the checkpoint's engine |
| the 9-candidate vocabulary values | **INSTRUMENT (ours)** — no primitive table is committed anywhere in `stack/`; the values are stated in the harness and held IDENTICAL across arms |
| the 100 ms budget, the 95.3 % agreement bar | **INHERITED** — carried from the runbook, not re-derived |
| four-family ADE/longitudinal/lateral/strategic deltas | ⛔ **NOT MEASURED HERE.** §4.1 establishes that the batch-9 engine is numerically the batch-1 engine, so the existing gate transfers; this run adds only the TACTICAL decision family (§5) |
