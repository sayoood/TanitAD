# Thor optimisation & productionisation — the execution plan (2026-08-02)

**Owner:** Production & Optimization (Saturday stream).
**Input:** the orchestrator's O1–O14 backlog in `THOR_DEPLOYMENT_RUNBOOK.md` §6, after the first
successful Thor deployment (272.56 → 98.63 ms measured, ≈51.2 ms projected with TRT-fp16).
**Purpose:** turn that list into an ordered, falsifiable, resource-matched programme — and record
the four things found while reading it that **change its order**.

> ⚠️ This plan does not re-litigate the Thor result. The 2.76× is measured and the toolchain work
> was excellent. What follows is about what the measurement **covers**, which is a different question
> from whether it is correct.

---

## 0. TL;DR — the four findings that re-order the backlog

| # | finding | evidence | consequence |
|---|---|---|---|
| **F1** | ⭐⭐ **The measured tick rolls ONE candidate. The deployed selector loops over NINE.** | `config.py:95` `n_maneuvers: int = 9  # (3 steer x 3 accel)`; `fourbrain.py:571` `for m in maneuvers:` … inner `for k in range(m.actions.shape[0])`; `thor_combined_tick.py:119-134` rolls a single `[1,W,·]` state | The 51.2 ms projection covers **encode + one candidate roll**. The deployed decision tick is **encode + M×K predictor steps + `step_readout` decode + scoring**. ⇒ **O9 (batched candidates) moves P2 → P0.** It is not a throughput nicety; it decides whether we are inside 100 ms at all. |
| **F2** | ⭐⭐ **The Thor recipe deploys bf16 on the encoder. Our standing precision policy forbids bf16 on the decision path — measured, twice.** | Thor: bf16 encoder rel-err **0.0059**, no decision metric. Ours (4060, 64 real windows, step-6500): bf16 → imagine-and-select agreement **67.2 %**, waypoint shift **47.7 cm mean / 3.58 m max**; fp16 → 95.3 %, 3.9 cm | The 6.76× — the single largest lever in the whole result — rests on a **numerics** rel-err on **random weights**, on a precision our own decision-space measurement rejected. ⇒ **the bf16 decision-agreement gate is P0, ahead of O2/O4/O5.** If bf16 fails, the encoder win must be re-earned in fp16 (which is what O2 is for). |
| **F3** | **O8 is already answered — close it. O4's headline saving is overstated ~2×.** | `thor_fullroll_graph.json`: full-roll graph **1.02×** vs per-step (falsifier was <1.1×) ⇒ **O8 FIRES, close.** `thor_ksweep.json` is an **eager** sweep (4.107–4.308 ms/step): K20→K10 saves **40.85 ms eager**, but at TRT-fp16 (1.168 ms/step) the same cut saves **11.7 ms**, not the runbook's "−23 ms at TRT speed" (that figure is the entire K20 TRT roll) | Two P1 slots free up. **O4's value drops to 11.7 ms of a 51 ms tick** — a poor trade against an imagination-horizon accuracy risk. **Demote O4 below O2/O9.** |
| **F4** | **The MHA retraction's scope is untested where the retracted claim was made.** | The bug is measured on **torch 2.13 / Thor**. The retracted claim was made on **torch 2.11 / dev box**, where parity measured 8.8e-6 / 1.2e-5 — i.e. it passed *there*. Fastpath fusion is **shape- and config-conditional**, so "2.13 regression" and "2.11 near-miss that our shapes happened to dodge" are both live | A 90-minute local experiment separates them, and either way we owe an **executable export guard** so no exporter can ever again emit `_native_multi_head_attention` silently. **This is the run's zero-dependency P0** — it needs no Thor, no pod, no credit. |

---

## 0.5 STATUS — B1 EXECUTED the same day (2026-08-03, Thor back online)

Full result: `Research/2026-08-03-thor-candidate-fan-and-engine-graph.md` ·
`Implementation/incoming/2026-08-03-thor-b1-fan/`.

| plan item | outcome |
|---|---|
| **B0** val data on Thor | ✅ **DONE — by the orchestrator, not by us.** `~/valdata/physicalai-val-0c5f7dac3b11` (2.4 GB `ep_*.pt`) + a `w120-256x640cyl` variant; `~/models/` now carries **flagship-v1-speedjerk, v5f, refc-base, refc-xl, rollout-recovery, flagship-v4.2b** ⇒ **B2/B3 are unblocked** |
| **B1** complete tick / **F1** | ✅ **CONFIRMED AND RESOLVED.** 9 candidates serialised through the shipped batch-1 engine = **243.84 ms (244 % of budget)**; through a **batch-9 engine (built + timed)** = **56.13 ms (56 %)**. ⇒ **rebuild the engine at batch 9 or dynamic — it is a deployment requirement worth 4.3×** |
| **F4** MHA fastpath | ⚠️ **DID NOT REPRODUCE at 6 cells** (predictor × opset 17/18 × fastpath ON/OFF; 10 MHA modules present; opset 18 fastpath-ON exports **clean**). ✅ But the question that mattered is answered: the **corrected-graph engine is 1.187 ms vs the published 1.168 ms (1.6 %)** ⇒ **the 5.33× survives** |
| **NEW — O2-pre** | 🔴 **O2 is blocked by a defect that is not TensorRT.** The encoder **does not export at 176×624**: `adaptive_avg_pool2d, output size that are not factor of input size`. The 2026-07-08 "encoder exports clean" claim was measured at **256×256** ⇒ **geometry-conditional and false for the shipping geometry.** Fix + an export test **at the deployed geometry** |
| **A1** local export guard | **re-scoped** — the guard is still owed, but its content changes: the real defects found are *geometry-conditional export failure* and *a wiring-bug-shaped parity error*, not a fastpath flag |
| next | **B2** (bf16-vs-fp16 decision agreement, trained ckpt, real windows) — now unblocked by B0 |

## 1. Resource reality — measured today, not assumed

| resource | state | evidence |
|---|---|---|
| ⛔ **Thor (`tanitad-thor`, 192.168.178.93)** | **OFFLINE** | three probes from the same subnet (dev box `192.168.178.198/WLAN`): `ssh` → *connection timed out*; ICMP → fail; TCP **22 and 80** → fail. An ARP entry (`f8-3d-c6-91-6e-22`) survives in cache, which is **not** liveness. ⇒ **PI action: power the board on.** |
| ⛔ **All four A40 pods** | **STOPPED** (credit $3.61) | `Project Steering/POD_SHUTDOWN_2026-08-02.md` |
| ✅ **RTX 4060 (dev box)** | available | torch **2.11.0+cu128**, onnx 1.22, onnxruntime 1.27 (CPU EP), ⛔ **no `tensorrt`** — P1.4a still open locally |
| ⛔ **val windows on Thor** | **absent** | LOOP_STATE 14:55 UTC: *"Remaining data gap: val episodes"*. Every Thor number to date is on **random weights** |
| ✅ **banked reference panel** | present, offline | `stack/experiments/pod-rescue-20260802/eval/root/taniteval/results/` — `fourfam_v1-lf19.json`, `hier_v1-lf19.json`, `v1-lf19.json` (n=418 windows / 19 episodes). Aggregates only, no per-window arrays |
| ✅ **weights** | reachable without a pod | HF `Sayood/tanitad-flagship-4b-phase0` (v1) and `Sayood/tanitad-rollout-recovery` (RR-20/RR-CTL/refc), both public |

**Planning consequence.** Tier A must be executable on the 4060 alone; Tier B must be *pre-written
and armed* so that the moment Thor powers on it runs unattended; Tier C names the PI asks explicitly.
"Blocked on Thor" blocks one tier, not the plan.

---

## 2. What the 51.2 ms actually covers — stated precisely

Both of our tick harnesses measure a **partial** decision tick, and they are partial in **different**
places. This is not an error in either; it is a coverage statement that has never been written down.

| harness | encode | imagination | selection | decode / score | heads |
|---|---|---|---|---|---|
| 4060 `latency_cnce_baseline.py` | `encode(1 frame)` | **1 step**, **batch 9** | — | — | — |
| Thor `thor_combined_tick.py` | `encode_window(8 frames)` | **20 steps**, **batch 1** | — | — | — |
| **deployed** `TacticalSelector.select` | `encode_window` | **M candidates × K_prim steps** | argmin over scores | `step_readout` per candidate | tactical/strategic decode |

⭐ **The good news is already measured, on the 4060: batching the candidate fan is FREE.**
`select_K9` (one `world.imagine` at **batch 9**) cost **5.69 ms** against `predict_1pass` at
**6.14 ms** — nine candidates for **0.93×** of one. The predictor is small-tensor/launch-bound, so
the batch dimension is nearly free. That is the mechanism O9 proposes, already demonstrated on other
silicon.

⚠️ **The engineering consequence nobody has flagged: the shipped `predictor_fp16.plan` is built at
batch 1.** TensorRT engines are shape-static unless an optimization profile says otherwise. A
9-candidate fan through that engine means **9 sequential enqueues** (≈9 × 1.168 ms × K), not one
batched call. ⇒ **the engine must be rebuilt with a batch-9 (or dynamic-batch) profile**, and that
rebuild belongs in the same experiment as the measurement.

**Worst case if the fan is serialised at K=20:** 27.8 + 9 × 23.36 ≈ **238 ms — 2.4× over budget.**
**Expected case if the fan batches like the 4060's:** ≈ 27.8 + ~25 ≈ **53 ms.** The spread between
those two is the largest open uncertainty in the deployment, and it is one experiment wide.

---

## 3. The plan

Each item: **goal · method · resource · expected number · falsifier · INSTRUMENT-FAIL branch ·
deliverable**. Falsifiers and the instrument-fail branch are pre-registered here, before any run
(GATE_PROTOCOL §0.3/§0.7 — C63's lesson: a prereg with no instrument-fail branch has nowhere to put
an out-of-range result).

### Tier A — executes now, 4060 only, zero external dependencies

#### **A1 · MHA export-parity re-verification at torch 2.11 + a permanent export guard** *(closes F4)*
- **Goal:** decide whether the 2026-07-08 "ONNX-clean" claim was *wrong then* or *broken since*, and
  make it impossible to re-break silently.
- **Method:** a 2×2×2 matrix on the dev box — fastpath {ON, OFF} × opset {17, 18} × geometry
  {256×256 legacy, 176×624 v5f deployed} — exporting the predictor and the encoder+readout, scoring
  ORT-vs-eager `max|Δz|` and rel-err in one process, plus a static scan of the exported graph for
  `_native_multi_head_attention`.
- **Resource:** 4060, ~90 min, $0.
- **Expected:** fastpath-OFF parity ≤1e-5 in all four cells. Fastpath-ON at opset 17: **either**
  ≥1e-1 (⇒ the 2026-07-08 claim was a shape-dependent near-miss and every artifact resting on it is
  suspect) **or** ≤1e-5 (⇒ genuine 2.13 regression; the retraction is correctly scoped to 2.13+).
- **Falsifier:** if fastpath-ON parity is clean at **both** geometries and **both** opsets, the
  "silently wrong at opset 17" claim does **not** generalise to 2.11 and must be labelled
  torch-2.13-specific in the runbook.
- **INSTRUMENT-FAIL:** if ORT-CPU and eager disagree with fastpath OFF (the control), the harness is
  wrong, not the exporter — stop and fix the harness before reading any cell.
- **Deliverable:** intake `2026-08-02-onnx-export-guard` — an `export_safe()` helper that sets
  `torch.backends.mha.set_fastpath_enabled(False)` and **refuses to return an exported graph
  containing `_native_multi_head_attention`**, + failing-then-passing tests, + the matrix JSON.
  G-P1 compliant (file:line + measured numbers).

#### **A2 · Production-compliance review #4 — `taniteval/` + `tanitad/eval/`**
- **Why this cluster, now:** it is the **only unreviewed cluster that decides every deployment gate**,
  and since 2026-08-02 the four families are *binding*. `PRODUCTION_READINESS.md` has empty rows for
  both. An instrument defect here silently invalidates O1, O4 and O5 at once — the C63 failure mode
  (an imported metric whose precondition was never measured) is this class.
- **Method:** the standard checklist against `four_families.py` (344 lines) and `tanitad/eval/`,
  with priority on: UNAVAILABLE-path honesty (does a missing `hier` really surface as UNAVAILABLE
  with reason+n, or can it be silently dropped?), the `MIN_DS_M` curvature exclusion counter
  (comparability across arms), NaN/empty-mask behaviour in `_masked`, determinism of the bootstrap
  seed, and batch-1/edge compatibility.
- **Resource:** dev box, CPU, $0.
- **Deliverable:** one small intake package with failing-then-passing tests.

#### **A3 · Arm Tier B — write and stage the Thor job cards *before* the board is up**
Not a report about being blocked: the actual runnable scripts, committed, so power-on → results with
no authoring latency. Job cards for B1/B2/B3 below, each self-contained
(`PYTHONPATH=/usr/lib/python3.12/dist-packages` for TRT bindings, `ssh -n` inside any piped
heredoc, `OMP_NUM_THREADS=6` before any multi-arm panel — all three are documented traps).

---

### Tier B — armed, fires the moment Thor is powered on

#### **B1 · ⭐ THE COMPLETE DECISION TICK, measured — supersedes O3, absorbs O9** *(closes F1)*
- **Goal:** the first end-to-end number that covers what actually ships.
- **Method:** time the full path — `encode_window` → **M=9 candidate fan** → K-step roll →
  `step_readout` decode → score/argmin → tactical+strategic head decode — in four configurations:
  (a) fp32 eager baseline, (b) bf16 encoder + CUDA-graph predictor, (c) + TRT-fp16 predictor at
  **batch 1, fan serialised**, (d) + TRT-fp16 predictor rebuilt at **batch 9 / dynamic profile**.
  p50 **and** p99, warmup 10, `cuda.synchronize()` around every region.
- **Expected:** (c) ≈ 238 ms (over budget); (d) ≈ 53 ms if the fan batches as it did on the 4060.
- **Falsifier:** if (d) > 1.1 × (encoder + one batched roll), the candidate fan does **not** amortise
  on Thor and the maneuver vocabulary itself becomes a deployment parameter (⇒ re-open O4/K and a
  vocabulary-size study, in that order).
- **INSTRUMENT-FAIL:** if (a) does not reproduce 272.56 ms ±10 %, the harness differs from the
  profile run — reconcile before reading (b)–(d).
- **Why first:** every other Thor item's cost/benefit is quoted against a tick that this measurement
  may move by 4×.

#### **B2 · ⭐ bf16-vs-fp16 encoder DECISION-agreement on Thor** *(closes F2)*
- **Goal:** decide whether the 6.76× lever is deployable at all.
- **Method:** trained checkpoint (v1 from HF; RR-20 as the second arm if time allows) on real val
  windows, fp32 reference vs bf16 vs fp16 encoder, scoring **imagine-and-select agreement** and
  **decoded-waypoint shift** — the same decision-space metrics that rejected bf16 on the 4060 — plus
  the four families with the paired episode-cluster bootstrap.
- **Bar (pre-registered, inherited and explicit):** agreement **≥95.3 %**, waypoint shift **≤4 cm**.
- **Falsifier:** bf16 below the bar ⇒ **bf16 does not ship**; the encoder win must be re-earned via
  **O2 (TRT-fp16 encoder engine)** and the 51.2 ms projection is void until O2 lands.
- **INSTRUMENT-FAIL:** if the fp32 Thor panel does not reproduce the banked A40 panel
  (`fourfam_v1-lf19.json`) inside its CI, **no precision verdict is admissible** — the disagreement
  is the platform or the data build, not the precision. This is the C63 precondition check, and it
  is a **prerequisite of B2, not an afterthought**.
- **Depends on B0 (below).**

#### **B0 · val-window cache built ON Thor, from HF — the unblocker for every accuracy item**
- **Goal:** remove the pod dependency that O1 silently carries.
- **Method:** pull the val episodes from the HF corpus directly to Thor (880 GB free, WiFi;
  overnight is fine), rebuild the window cache with the repo's own builder, and **verify the parity
  sha against the committed manifest** before any arm runs. Weights from public HF repos.
- **Falsifier:** if the parity sha does not match, **stop** — a re-selected val set breaks
  cross-arm comparability and is refusable under the parity invariant.
- **Note:** this is the piece that makes O1 a three-step project (data → harness validation →
  verdict) rather than the single experiment the runbook lists.

#### **B3 · O1 proper — four-family accuracy gate on the TRT-fp16 engine**
Runs after B0 + the B2 instrument check. Engine vs eager, **paired** episode-cluster bootstrap on
the same windows, all four families **per-family, never pooled**, each with its estimator, CI, and —
where a family cannot be computed — the reason and the n. Falsifier as the runbook states: any
family degrading beyond CI, or agreement < 95.3 %, ⇒ **fp16 does not ship**.

---

### Tier C — after B1/B2 report, in the order their results imply

| item | status after this plan | rationale |
|---|---|---|
| **O2** TRT engine for the encoder | **P0 if B2 rejects bf16**, else P1 | It is the fallback that makes the encoder win survive a bf16 failure — and it is the larger stage again either way (27.8 of 51.2 ms). |
| **O5** INT8 PTQ | P1, **gated on B3** | Needs a working four-family gate to be judged at all; running it before B3 produces an unreadable number. |
| **O4** K 20→10 | **demoted** (was P1) | F3: the true TRT-speed saving is **11.7 ms**, not 23, against an imagination-horizon accuracy risk. Poor trade until the tick is known (B1) — and if B1 comes in at 53 ms we may not need it. |
| **O7** `nvpmodel` power modes | P1, cheap | 1976 mW / 61.9 °C says the envelope is unexplored; independent of every accuracy gate; a genuinely free parallel item while data downloads. |
| **O6** NVFP4 · **O14** 2:4 sparsity | P2, **evidence already against** | Measured: FP8 gives 1.21× at our 8×2048 shape vs 1.97× at 4096³. Our tensors are on the wrong end of the curve. ⚠️ **B1(d) may move them**: a batch-9 fan is a bigger GEMM, so re-price both *after* B1, not before. |
| **O8** one engine for the whole roll | ⛔ **CLOSE** | Measured 1.02× vs per-step; its own falsifier (<1.1×) fires. ⚠️ Narrow caveat for the record: this was tested at **CUDA-graph** level, not TRT level; the TRT variant inherits a strong prior but is untested and is not worth a slot. |
| **O10–O13** resolution, multi-camera, Orin port, DLA | P2, unchanged | All are downstream of a known tick and a working accuracy gate. |
| **P1.4a** local TRT toolchain | P2, **downgraded** | Thor now *is* the TRT box. The dev-box install stops being on the critical path; keep it as a Tools&DevEnv convenience ask, not a blocker. |

---

## 4. Sequencing

```
NOW (no dependencies)          A1 export guard ──┐
                               A2 eval-cluster review ──┤→ intake packages
                               A3 arm the job cards ────┘

PI: power on Thor ─→ B0 val cache (overnight)
                        │
                        ├─→ B2 instrument check (Thor fp32 vs banked A40 panel)  ← MUST pass first
                        │        └─→ B2 bf16-vs-fp16 decision agreement ──→ decides O2's priority
                        └─→ B1 complete tick (needs no val data — runs in parallel, random weights OK
                                               for LATENCY; ⛔ never for accuracy)
                                 └─→ re-price O4 / O6 / O14 against the real tick
                                         └─→ B3 = O1 four-family engine gate ──→ O5 INT8
```

⭐ **B1 needs no val data.** Latency is a property of shapes and kernels, not of weights — so the
complete-tick measurement can run the same hour the board comes up, while B0 downloads. That
ordering is deliberate: it puts the highest-uncertainty item first at zero data cost.

---

## 5. What this plan will not do, and why

1. **Not re-run `torch.compile` on any device.** Failed on the 4060 (no Triton) and on Thor
   (InductorError). Two platforms, two causes — settled. Manual capture + TensorRT.
2. **Not quote the 51.2 ms as "the tick"** in any downstream document until B1 reports. It is
   *encode + one candidate roll*, and the plan says so in §2.
3. **Not run O5/INT8 before B3.** An accuracy-critical quantisation without a working four-family
   gate produces a number nobody may act on.
4. **Not inherit the 95.3 % bar silently.** It comes from the 4060 at 256×256 on step-6500 v1. It is
   the right bar to *pre-register*, but B2/B3 must state that its provenance is a different geometry
   and a different checkpoint.

---

## 6. PI asks

1. ⭐ **Power on the Thor** (`192.168.178.93`) — measured offline at three layers today. It is the
   only blocker on Tier B, and Tier B contains both decision-grade experiments.
2. **Credit decision on the pods** — *not* needed for this plan. Every item here runs on the 4060 or
   the Thor. Recorded only so the plan's independence from that decision is explicit.
3. **For information:** the Jetson factory-default password is still live on a WiFi-connected board
   (flagged 2026-08-02 15:0x); worth changing before the board is exposed further.

---

## 7. Evidence class

| claim | class |
|---|---|
| Thor latencies, rel-errs, K-sweep, full-roll graph, TRT build/gate | **MEASURED (not ours — Architecture & Inference, 2026-08-02)**, `…/incoming/2026-08-02-thor-deployment-profile/*.json`. Re-derived here from the raw JSON, not from prose |
| `n_maneuvers = 9`; the selector's per-candidate loop; the 1-candidate harness | **MEASURED (ours, source read)** — `config.py:95`, `fourbrain.py:562-585`, `thor_combined_tick.py:119-157` |
| batch-9 imagine is free (0.93× of batch-1) | **MEASURED (ours, 4060)** — `latency_cnce_baseline.py:71-80`, run #1 JSON |
| bf16 decision-unsafety (67.2 % agreement, 47.7 cm) | **MEASURED (ours, 4060, 64 real windows, step-6500)** — `half_precision_step6500.json`. ⚠️ **different geometry and checkpoint from Thor** — which is precisely why B2 must re-measure rather than transfer it |
| Thor offline; pods stopped; no local `tensorrt` | **MEASURED (ours, today)** — 3-layer probe from the same subnet; `POD_SHUTDOWN_2026-08-02.md`; venv import probe |
| "O4 saves −23 ms at TRT speed" | ⛔ **CORRECTED** — the ksweep is an eager sweep (4.107–4.308 ms/step); at TRT-fp16 (1.168 ms/step) K20→K10 saves **11.7 ms**. 23.36 ms is the *entire* K20 TRT roll |
| serialised-fan 238 ms / batched-fan 53 ms | **ESTIMATED** from measured per-step costs — B1 exists to replace both with a measurement |
| torch-2.11 MHA export status | **UNKNOWN** — A1 exists to settle it. The 2026-07-08 "no unexportable ops" claim is **RETRACTED for 2.13** and **unverified for 2.11** |
