# Thor B1 — the candidate fan is the whole deployment question, and it is now answered

**Date:** 2026-08-03 (Europe/Berlin; Thor's clock, `00:14`–`00:3x`).
**Hardware:** NVIDIA Thor (Blackwell sm_110), aarch64, torch **2.13.0+cu130**, TensorRT **10.13.3.9**,
`~/venvs/tanitad-edge`. **Cost $0** (owned hardware). Wall-clock ≈ 25 min including two engine builds.
**Plan ref:** `THOR_OPTIMISATION_PLAN.md` §3 Tier B / **B1** (findings **F1** and **F4**).
**Artifacts:** `Implementation/incoming/2026-08-03-thor-b1-fan/` — harness + raw JSON.
**Model:** `WorldModel(flagship4b_config)` at v5f deployed geometry **176×624 / 117° cylindrical**,
window 8, state_dim 2048, action_dim 3. ⚠️ **Random weights — this is a LATENCY run.** Latency is a
property of shapes and kernels, not of weights; **no accuracy claim is made or implied here.**

---

## 1. The headline

⭐⭐ **The 9-candidate fan is the difference between 244 % and 56 % of the 100 ms budget — and the
fix is a one-line change to how the TensorRT engine is built.**

| deployed shape | tick | vs 100 ms budget |
|---|---|---|
| published 1-candidate projection (what the runbook reports) | 53.98 ms | 54 % |
| ⛔ **9 candidates serialised through the shipped batch-1 engine** | **243.84 ms** | ⛔ **244 %** |
| ✅ **9 candidates through a batch-9 engine** (built + timed in this run) | **56.13 ms** | ✅ **56 %** |

**F1 is confirmed and resolved in the same run.** The deployed `TacticalSelector` loops over **9**
maneuvers (`config.py:95`, `fourbrain.py:571`) while every published Thor tick rolled **one**. Left
as-is — a 9-candidate fan pushed through the `predictor_fp16.plan` that exists today, which is
**batch-1 static** — the deployment misses its deadline by 2.4×. Rebuilt at batch 9 it lands at
56 ms with 44 % headroom, only ~2 ms above the published 1-candidate projection.

---

## 2. The measurements

### 2.1 The fan amortises — but far better in TensorRT than in eager

**Eager predictor, one step, by batch** (p50 over 50 iters, warmup 10, `cuda.synchronize()` per rep):

| batch | p50 ms | p99 ms | **per candidate** |
|---|---|---|---|
| 1 | 4.214 | 4.866 | 4.214 |
| 3 | 6.478 | 7.284 | 2.159 |
| 5 | 10.352 | 11.374 | 2.070 |
| **9** | **11.117** | 11.874 | **1.235** |

⇒ nine candidates cost **2.64×** one candidate, i.e. **3.41× cheaper per candidate**.

⚠️ **This is materially worse than the 4060's prior**, where a batch-9 `imagine` cost **0.93×** of
batch-1 — nine candidates for less than the price of one. Thor amortises the fan, but **not for
free**. That difference is consistent with Thor's other batch finding (its encoder is compute-bound
where the 4060's was not) and is exactly why the plan required this to be measured rather than
transferred.

**Full K=20 roll:**

| roll | p50 ms |
|---|---|
| 1 candidate | **81.72** |
| 9 candidates, **batched** | 208.01 |
| 9 candidates, **serialised** | 728.77 |

⇒ batching the fan in eager saves **3.50×**. ✅ **Harness validation:** the 1-candidate roll
reproduces the published 81.5–81.7 ms to **0.02 %**, so this harness is measuring the same thing the
profile run measured.

### 2.2 TensorRT amortises the fan almost perfectly — and its edge grows with batch

fp16 engines built and timed in this run (`trtexec`, 200 iters, 500 ms warmup, median GPU compute):

| engine | median ms | per candidate | vs eager same batch |
|---|---|---|---|
| batch 1 | **1.187** | 1.187 | 3.55× |
| **batch 9** | **1.294** | **0.144** | ⭐ **8.59×** |

⭐ **Nine candidates for 1.09× the time of one.** Where eager pays 2.64× for the fan, the engine pays
**1.09×** — so the TRT advantage rises from 3.55× at batch 1 to **8.59×** at batch 9. The fan is
where TensorRT earns its toolchain, and that is not visible in any batch-1 measurement.

### 2.3 Reconstructed tick

Encoder bf16 measured **30.24 ms** this session (published 27.78; **+8.8 %** — the board was running
a desktop session with a browser during this run, so treat 27.8 as the clean-box figure and 30.2 as
the contended one; the fan conclusions are ratios and are unaffected).

```
1 candidate    : 30.24 + 20 × 1.187          =  53.98 ms   (54 % of budget)
9 serialised   : 30.24 + 20 × 1.187 × 9      = 243.84 ms   (244 %)  ⛔
9 batched      : 30.24 + 20 × 1.294          =  56.13 ms   (56 %)   ✅
```

⚠️ **Still a partial tick.** It covers encoder + imagination fan. It **excludes** `step_readout`
decode, candidate scoring, and the tactical/strategic head decode. It is nevertheless the first
number that covers the **candidate dimension**, which every previous tick omitted.
⚠️ **K=20 per candidate is an upper bound** — the selector rolls `m.actions.shape[0]` steps, the
maneuver-primitive length. The roll is linear in K (measured: 4.107–4.308 ms/step across K=4…20), so
a shorter primitive scales this down proportionally.

---

## 3. ⛔ A RETRACTION-CLASS FINDING: the MHA-fastpath mechanism did not reproduce

The runbook's §3 records that exporting at **opset 17 with the MHA fastpath ON** produces a
**silently wrong graph (rel-err 0.726)**, and makes
`torch.backends.mha.set_fastpath_enabled(False)` mandatory. That claim also **retracted** our
2026-07-08 "ONNX-clean" result. Plan item F4 existed to scope it.

**Measured here, same device, same torch 2.13, same model, opset 17, predictor:**

| export | nodes | fused MHA op in graph | ORT-CPU rel-err vs eager | engine median |
|---|---|---|---|---|
| fastpath **ON** | 1223 | **no** | **3.6e-07** | 1.154 ms |
| fastpath **OFF** | 1223 | **no** | **3.7e-07** | 1.187 ms |

**The flag changed nothing** — identical node counts, parity clean **with the fastpath ON**, and a
latency difference of 1.6 %. The 0.726 error did not reproduce.

**The competing explanation, and why it is the stronger one.** The artifact carrying the 0.726 is
`thor:~/thor_trt_accuracy.json` (written 18:13, **superseded** by `thor_trt_gate.json` at 18:19 and
never committed). It reports **fp32 0.72824** and **fp16 0.72818** — near-identical across
precisions. The runbook's own **learning #8** names that exact signature: *"identical error across
precisions is the signature of a wiring/test bug, not a precision problem"*, and its learning #9
records that the first gate attempt did compare an engine against a **different random model**.
⇒ **HYPOTHESIS (well-supported, not yet closed): the 0.726 was that wiring bug, and the fastpath was
misattributed as its cause** — inside the very document that warns about this failure mode.

⛔ **What this does NOT license.** Absence at one location is not absence: the probe above is one
tower (predictor), one opset (17), one torch (2.13). A second probe — opsets 17 **and** 18, fastpath
ON **and** OFF, on **both** the predictor and the encoder, plus a census of the actual
`nn.MultiheadAttention` modules in each tower — is running as `thor_b1b_fastpath_probe.py`; §5
records its verdict. **Until it lands, neither the runbook's claim nor this one is settled**, and
`set_fastpath_enabled(False)` stays in every export path — it costs 5.1e-7 in eager and 1.6 % here,
which is cheap insurance either way.

✅ **What is settled regardless: the published 1.168 ms is admissible.** This run reproduced it at
**1.154 ms** from the same (fastpath-ON) export path and got **1.187 ms** from the corrected one —
**1.6 % apart**. So the **5.33× and the 51.2 ms tick survive the correction**, which was the open
risk F4 flagged. They were, however, published from an export path whose correctness was
unverified at the time; this run is what makes them admissible.

---

## 4. What this changes

1. ⭐ **The batch-9 (or dynamic-batch) predictor engine is now a deployment requirement, not an
   optimisation.** `predictor_fp16.plan` as shipped is batch-1 static. Ship it and the fan
   serialises at 244 % of budget.
2. **O9 was correctly promoted.** It is not "P2 structural" — it is worth **4.3×** on the tick.
3. **The eager fallback is not a fallback.** If the engine is unavailable, the eager batched fan
   costs 208 ms of roll — already over budget before the encoder. The engine is load-bearing.
4. **O4 (K 20→10) drops further.** At the batch-9 engine's 1.294 ms/step the whole K=20 fan is
   25.9 ms; halving K saves **12.9 ms** of a 56 ms tick. Against an imagination-horizon accuracy
   risk that is a poor trade, and we now have 44 % headroom without it.
5. **Re-price O6/O14 (NVFP4, 2:4 sparsity) — the argument against them weakened.** They were parked
   because our GEMMs are too small; batch 9 makes them 9× wider in the batch dimension, and the
   engine's batch-9 efficiency (8.59× over eager) shows this shape is far better fed. Still P2, but
   the "wrong end of the size curve" reasoning no longer applies unchanged.

## 5. B1b — the second probe, and the encoder blocker it uncovered

**Module census first** (because "we have no MHA" would have been the boring explanation, and it is
false): `nn.MultiheadAttention` instances — **predictor 10, encoder 12, whole model 41**. The modules
the mechanism needs are present.

| tower | opset | fastpath | export | fused MHA op | ORT rel-err |
|---|---|---|---|---|---|
| predictor | 17 | **ON** | ✅ | **no** | **4.32e-07** |
| predictor | 17 | OFF | ✅ | no | 4.41e-07 |
| predictor | **18** | **ON** | ✅ | **no** | **4.32e-07** |
| predictor | 18 | OFF | ✅ | no | 4.41e-07 |
| encoder | 17 | ON | ⛔ **fails** | — | — |
| encoder | 17 | OFF | ⛔ **fails** | — | — |

### 5.1 The fastpath mechanism does not reproduce on the predictor — at either opset

Four cells, 10 MHA modules present, and **nothing changes**: no graph carries a fused MHA op, parity
is 4.3e-07 with the fastpath **ON**, and — the sharpest cell — **opset 18 with the fastpath ON
exports cleanly**, where the runbook records it *"fails loudly:
`aten::_native_multi_head_attention` unsupported"*. The plausible reason is mundane:
`torch.onnx.export` traces under conditions in which the fastpath does not engage at all, so the
flag is a no-op on this path.

⚠️ **Fairness caveat, and it is a real one:** the script that produced `thor_trt_gate.json` is **not
on disk** — only my two scripts reference `set_fastpath` anywhere on the box, so that run was a
transient heredoc. **I cannot inspect what it actually exported**, so I cannot exclude that it
wrapped a different module. My claim is therefore scoped to what I ran: *at 6 probed cells on
torch 2.13, this model, the mechanism does not appear.*

**Combined with §3, the reading is:** the 0.726 that produced the retraction has (a) the
near-identical-across-precisions signature the runbook itself calls a wiring bug, (b) a sibling in
the same session that was *exactly* that (learning #9: a gate compared an engine to a **different
random model**), and (c) no reproduction across 6 cells. ⇒ **the retraction of our 2026-07-08
"ONNX-clean" claim rests on a mechanism that does not reproduce, and should itself be revisited by
its owner.** Recorded in the runbook as a measured annotation, not a deletion.

⛔ **Keep `set_fastpath_enabled(False)` in every export path regardless.** It costs 5.1e-7 in eager
and 1.6 % of engine latency here. Cheap insurance against a mechanism we have not fully mapped is
the right call even when it looks inert.

### 5.2 ⭐ NEW BLOCKER — the encoder does not export at the deployed geometry

Both encoder cells failed **before** MHA could matter, and for a reason nobody has recorded:

```
SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
output size that are not factor of input size
```

**Why this matters and why it was invisible until now:**

1. ⛔ **O2 (TRT engine for the encoder) cannot start today.** It is the designated fallback if the
   bf16 decision-agreement gate (B2) rejects bf16 — i.e. the fallback for the single largest lever
   in the whole Thor result — and it is blocked by an unrelated, unreported export defect.
2. 🔴 **It retracts an inherited claim, again on geometry.** Our 2026-07-08 row says *"encoder+readout
   export clean at opset 17 AND 18"* — measured at **256×256**, where the adaptive-pool output size
   evidently *is* a factor of the input. At the deployed **176×624** it is not. ⇒ the claim is
   **geometry-conditional and false for the shipping geometry**. Same root-cause class as the
   torch-version lesson: *a passing export check re-asserted on a changed configuration without
   re-running it*.
3. ✅ **The fix is likely small and belongs in `stack/`:** replace the `adaptive_avg_pool2d` call with
   an explicit-kernel `avg_pool2d` (or pad to a factor) wherever the encoder's pooling is
   shape-derived. That is an intake package with a failing-then-passing export test — and the test
   must run **at the deployed geometry**, which is the whole lesson.

⇒ **New backlog item O2-pre, ahead of O2.** Filed against the plan.

---

## 6. Evidence class

| claim | class |
|---|---|
| every latency, node count, ORT parity, engine build in §2–§3 | **MEASURED (ours)** — `2026-08-03-thor-b1-fan/thor_b1_fan_and_fastpath.json`, Thor, torch 2.13.0+cu130, TRT 10.13.3.9, random weights |
| `n_maneuvers = 9`; the per-candidate selector loop | **MEASURED (ours, source read)** — `config.py:95`, `fourbrain.py:562-585` |
| 4060 batch-9 prior (0.93×) | **MEASURED (ours, earlier)** — `latency_cnce_baseline.py:71-80` |
| published 27.78 ms encoder / 1.168 ms engine / 81.5 ms roll | **MEASURED (not ours)** — Architecture & Inference `2026-08-02-thor-deployment-profile/*.json`; both reproduced here (roll to 0.02 %, engine to 1.2 %) |
| "the 0.726 was a wiring bug, not the fastpath" | **HYPOTHESIS, now strongly supported** — non-reproduction at **6 cells** (predictor × opset 17/18 × fastpath ON/OFF, plus 2 encoder cells that fail earlier), with 10 MHA modules present. ⚠️ Not closed: the original gate script is not on disk, so its exact export target is uninspectable |
| encoder fails to export at 176×624 on `adaptive_avg_pool2d` | **MEASURED (ours)** — `thor_b1b_fastpath_probe.json`, both fastpath settings, opset 17 |
| the 56.13 ms deployed tick | **MEASURED per stage, COMPOSED arithmetically** — and it excludes decode/scoring/heads; a wired end-to-end selector measurement is the follow-up |
| anything about accuracy | ⛔ **NOT ESTABLISHED HERE** — random weights. The four-family gate (plan B2/B3) is unaffected by this run and still owed |

## 7. Next

1. **B2 — bf16-vs-fp16 encoder decision-agreement** on real windows with a trained checkpoint.
   **Now unblocked:** Thor has `~/valdata/physicalai-val-0c5f7dac3b11` (2.4 GB, `ep_*.pt`) and
   `~/models/flagship-v1-speedjerk`, `~/models/v5f`. Plan item **B0 is DONE** — the orchestrator
   landed the data.
2. **B3 (= O1)** — the four-family gate on the engine, after B2's instrument check reproduces the
   banked A40 panel.
3. **Wire the real selector** for a true end-to-end tick (adds decode + scoring + heads to §2.3).
