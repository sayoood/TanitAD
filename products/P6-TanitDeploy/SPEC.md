# P6 — TanitDeploy · product specification

`Owner: TanitAD_DeployFlyWheel (AGENT_CHARTERS.md §4). Created 2026-08-23.
Governed by TANITAD_PROGRAMME.md; where this doc disagrees with the programme or
MODEL_REGISTRY.md, THEY win. Terms are VOCABULARY.md terms.`

## 0. What TanitDeploy is

**A checkpoint becomes a deployment when it runs on the target inside its
latency, memory and power budget AND its accuracy delta is measured.** Thor
first. Everything else in this document serves that sentence.

⛔ **THE PRODUCT'S ONE INVIOLABLE RULE (charter §4):**
> **A quantization without a paired eval is not a deployment.**

A latency number alone is a *profile*. A profile plus a paired four-family eval
inside a pre-registered budget is a *deployment*. TanitDeploy never ships the
first while calling it the second, and never quotes a speedup without its
accuracy number in the same sentence.

## 1. Scope

**In:** export (torch→ONNX→TensorRT, and torch-native graph/compile paths) ·
the quantization ladder · profiling and its admissible probes · the optimisation
playbook · engine provenance and rebuild recipes · the deployment gate.

**Out:** what the model should *be* (Master Mind) · what "good" means (EvalFlyWheel
owns the criteria; P6 *runs* them, never redefines them) · training (P4).

## 2. Targets

| target | status | notes |
|---|---|---|
| **Jetson Thor** (`thor6`, sm_110, aarch64) | **PRIMARY** | torch 2.13.0+cu130, TensorRT **10.13.3.9+cuda13.0**, Triton 3.7.1 present. ⚠️ single GPU in the fleet — everything queues behind it |
| **dev-box RTX 4060** (8.6 GB, Windows) | secondary | ⛔ **no Triton** ⇒ `torch.compile`/inductor unusable; manual CUDA graphs only. Never pool its numbers with Thor's |
| A40 / A6000 pods | ⛔ **RETIRED** | all terminated. Every pod-era benchmark is orphaned by host; the numbers survive as INHERITED, the scripts do not run |

### 2.1 Thor's inverted instincts (MEASURED — do not re-derive)

- ⛔ **Only `torch.cuda.max_memory_allocated()` is admissible.** `mem_get_info`,
  `free`, `tegrastats` and `VmRSS` all lie on unified memory, **in both
  directions** (2026-08-03: 3.4 GB "free" with 60 GB allocated and written).
- ⚠️ **But device-allocated is not host footprint.** Sizing a deployment needs
  RSS: 11.48 GB after load vs 1.2 GB `max_memory_allocated` (`thor_profile.json`).
  Two questions, two probes — name which one you are answering.
- ⚠️ **Throughput saturates early.** 20 SMs. Measured for *inference* 2026-08-23:
  b4→b8 buys 7–21 % for ~74 % more memory. **Deploy default: batch ≤ 4.**
- ⚠️ **Latency is launch-bound at batch 1**, not compute-bound. See §5.

## 3. The export ladder

| rung | path | when | state 2026-08-23 |
|---|---|---|---|
| **E0** | torch eager | always — the reference arm | ✅ |
| **E1** | **manual CUDA graphs** (`torch.cuda.CUDAGraph`) | first lever everywhere; **bit-identical**, needs no eval | ✅ **v7-tiny 1.66× @b1, all arms bit-identical** |
| **E2** | `torch.compile` | Thor only (Triton present); ⛔ never the dev box | ⛔ **BLOCKED on Thor** — all arms die on `Python.h: No such file` (`python3-dev` absent). One `apt` line away; backlog D-04a |
| **E3** | torch→ONNX (opset 17, fastpath **OFF**) | prerequisite for TRT | ✅ predictor + encoder both export |
| **E4** | ONNX→TensorRT via **`trtexec`** | the production path | ✅ engines built; ⛔ **python `tensorrt` bindings are NOT importable from either venv** — `trtexec` CLI is the only route, and that shapes every tool |

⛔ **A `.plan` is not a portable artifact.** It is bound to the device
architecture and the TensorRT build. Engines are **never** committed; what is
committed is the **rebuild recipe + descriptor + sha256** (§7).

## 4. The quantization ladder — each rung needs its paired eval

| rung | status | evidence |
|---|---|---|
| **fp32** | reference | — |
| **CUDA graph** | ✅ **free** — bit-identical output, no eval required | `2026-08-23-v7tiny-baseline-profile/` |
| **TF32** (at graph capture) | ⛔ **lossy** — +22 % but perturbs every learned tensor; gate required | `2026-08-23-thor-lever-ladder/` |
| **fp16 / bf16 (autocast)** | ⚠️ **conditionally open** | paired gate on flagship-v1 fired **F_LONGITUDINAL**; bf16 decision-agreement never run (G5) |
| **TRT-fp16 engine** | ⚠️ ships only past the gate | `predictor_v1_intent_dyn1-9_fp16` passes decision-agreement (1.0000 @K4, 0.9850 @K20, max regret 3.8 mm) but the four-family gate fires |
| **INT8** | ⛔ **TRIED AND LOST** | W+A INT8 ADE +0.021463 m (falsifier 0.02 **fires**) *and* 2.1 % **slower** on the predictor than fp16. Root cause localised: `SpatialGridReadout.proj` collapses to cosine **0.566** while every transformer block stays ≥0.9999 |
| **FP8 / NVFP4 / 2:4 sparsity** | ⛔ **never attempted** | backlog D-09 |

⚠️ **INT8's failure is a REPAIRABLE, LOCALISED failure, not a dead end.** One
module broke. Mixed-precision INT8 keeping the readout projection in fp16 is
untested and is the cheapest remaining accuracy-per-byte lever.

### 4.1 The gate every rung must pass

Run by the EvalFlyWheel's instrument, never a P6 re-implementation:

1. **Paired**, same windows both arms, episode-cluster bootstrap.
   ⛔ `overlapping_holdout_se` is forbidden.
2. **All four families** — LONGITUDINAL · LATERAL · TACTICAL · STRATEGIC — never
   pooled, each with its `n`. A family that cannot be computed is reported
   **UNAVAILABLE with its reason**, never silently dropped.
3. **A negative control that can fire** — an arm known to differ must be shown to
   separate, or the gate's PASS means nothing.
4. **Decision-space agreement**, not only regression error. Numerics rel-err is
   *not* a proxy: the same engine that reads 3.7e-4 rel-err costs **3.81 m** mean
   regret when its intent token is dropped.
5. ⛔ **A materiality threshold pre-registered BEFORE the run.** Its absence is
   why the current best config has no shipping verdict (§8, G2).

## 5. The optimisation playbook (ordered by measured return)

1. **CUDA graphs first.** Bit-identical ⇒ zero accuracy risk ⇒ no eval needed.
   Biggest win where the fixed cost dominates: **1.66× at b1**, 1.09× at b8.
   *Diagnostic*: fit `t(B) = a + b·B`. A large `a` is launch overhead. On v7-tiny
   `a` = 7.45 ms (R² 0.99999) and graphs removed 70 % of it.
2. **Batch to 4, not to fill memory.** Past 4 the SMs are saturated.
3. **TF32 at graph-capture time** — **+22 %** on top of graphs (6.10 ms vs 7.86 ms
   at b1, within-process paired A/B, both rounds). ⛔ **But it is NOT free**: it
   moves every learned tensor, *more* than fp16 does at the trunk (z_op
   5.61e-02 vs 2.70e-02), while leaving the plan output at exactly zero. It goes
   through the §4.1 gate like any other lossy rung.
   ⚠️ **The general lesson, and it is the expensive one**: the tensor that is
   easiest to screen ranked TF32 and fp16 in the OPPOSITE order to the tensors
   that matter. **Screen the LEARNED tensors, never only the final output.**
4. **Precision only after 1–3**, and only through the §4.1 gate. On small models
   fp16 can be **slower** (v7-tiny b1: 1.21× *slower* than fp32).
5. **Pin the kinematic integrator to fp32.** ⭐ MEASURED: `unicycle_rollout`
   inherits the autocast dtype and accumulates 60 steps of distance in it —
   0.67 ULP (fp16) / 4.60 ULP (bf16) at ~65 m. Upcasting the controls before the
   rollout is **bit-identical** to fp32 and costs ~nothing.
6. **TRT engine for the predictor** once the gate is passed.
   ⛔ **And the caller must batch** — a batch-9 engine driven by a serialised loop
   measures *worse* than batch-1 (272.8 vs 265.7 ms).
7. **Shared encoder across the candidate fan.** Re-encoding per candidate costs
   +5.6 ms @K8.

## 6. Profiling standard (binding)

Every P6 profile carries, or it is not a P6 profile:

- **n printed**, median **and** p90, warmup discarded and stated.
- **Memory from the admissible probe only**, named in the artifact.
- **Controls that must read a known value**, each able to fail:
  *timer resolves a definitionally-faster op · memory probe moves for a known
  allocation · fp32 repeat is bit-identical · param count equals the run's ·
  state-dict loads with zero missing/unexpected keys.*
  ⛔ **A failed control voids the run** — the output is a RESULT saying so, not a
  table with a caveat.
- **Reference-scale alongside every deviation** — an absolute error is
  uninterpretable without the magnitude it sits on, and **the statistic is
  named** (max-abs vs max-rel invert conclusions when references approach zero).
- **A rebuild recipe** with the exact command.

## 7. Engine provenance

Every engine has a descriptor JSON (source ckpt + step + shapes + `trtexec`
command + verify block) and an entry in `products/P6-TanitDeploy/engines/
THOR_ENGINE_REGISTRY.md` carrying its **sha256**. Binaries stay on the box;
recipes and hashes live in git.

⛔ **An engine built from a randomly-initialised model is not a deployment
artifact** — `~/trt/predictor_fp16.plan` is exactly that and is retained only for
provenance. ⛔ **An engine without the `intent` input computes the unconditioned
prediction** and the runtime must raise, never silently drop the token.

## 8. Success criteria for P6

| # | criterion | state |
|---|---|---|
| S1 | Every shipped config has a paired four-family eval inside a **pre-registered** budget | ⛔ **blocked — no materiality threshold ratified (G2)** |
| S2 | Latency/memory from admissible probes, `n` and p90 always | ✅ enforced by §6 |
| S3 | Every engine rebuildable from a recorded recipe | ⚠️ recipes exist for Thor's engines; registry created 2026-08-23 |
| S4 | A tick measured at the **deployed geometry**, not a proxy | ⛔ every end-to-end tick is at 256×256; 176×624 is `PROJ` only |
| S5 | The optimised path has a **production caller** | ⛔ `batch_fan=True` + `TRTPredictor` are wired into no driver |
| S6 | Deployment reproducible on a second box | ⛔ Thor is the only GPU; the 4060 cannot run the four-family gate (no val windows locally) |

## 9. Interfaces

**Consumes** checkpoints from P4 Training (+ `MODEL_REGISTRY.md` for every model
fact). **Supplies** EvalFlyWheel with configs to gate and leaderboard rows.
**Consumes** Lab findings from *Deployment & Optimization*. **Escalates**
integration to the Master Mind — never by leaving a note in a README.
