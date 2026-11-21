# COMMS — `2026-08-30-b1-gate-strongly-typed`

## Integration status: ⛔ ESCALATED — register row NOT applied by me

`Project Steering/GOALS_AND_CLAIMS.md` moved under this run (modified 09:15:06). Paste-ready
delta below. **Owner to apply: TanitAD Master Mind; implementation owner: DeployFlyWheel.**

---

## Delta — RE-SCOPE `D-B1-GATE` from a platform blocker to a build-path defect

The row currently reads as though the Thor FP8 path is unusable pending #4590. Proposed
addition (the existing content stays; #4590 is **re-verified OPEN**, so nothing is retracted):

> ⭐⭐ **RE-SCOPED 2026-08-30 — #4590 IS A DIAGNOSTICS BUG, NOT "FP8 IS BROKEN ON THOR", AND THE
> GATE IS UN-BLOCKABLE TODAY.** TensorRT docs, verbatim: *"FP8 quantization exclusively supports
> explicit quantization via Q/DQ layers. There is no calibration-based implicit quantization path
> for FP8 — implicit quantization is limited to INT8 only."* ⇒ the #4590 reporter set
> `BuilderFlag.FP8/FP4` on a graph with **no Q/DQ nodes**, so nothing was ever going to be
> quantised; **the genuine defect is the missing warning.** Corroborated on our own hardware by
> TRT issue **#4599**: FP8 **did** engage on Thor after ONNX Q/DQ surgery (layer info confirmed,
> ~20 % latency win on ViT-Base). ⇒ **our exposure is "we build via builder flags at all", not
> the platform.** Fix available now on **TRT 10.16.2 (JetPack 7.2.1, 2026-08-11)**: **ModelOpt
> explicit Q/DQ + `trtexec --stronglyTyped`** — which is **NVIDIA's own recommendation for Thor**
> (they measured INT8 +37 % on Orin vs **+2.7 % on Thor** and prescribed exactly this).
> ⛔ **TRT 11 would remove the bug class by construction (all per-precision builder flags
> deleted) but JetPack forbids it:** *"NVIDIA JetPack is not supported in TensorRT 11.2.1. Jetson
> deployments must remain on a TensorRT 10.x release."*
>
> ⛔ **AND THE GATE-DESIGN CONSEQUENCE: a numerical check can NEVER catch this failure** —
> comparing an FP32 reference against a *secretly-FP32* engine yields **zero error**, so the
> failure makes the gate score **better**. ⇒ **stage 1 must be STRUCTURAL**: plan-file size vs the
> FP16 baseline (#4590's own tell: **199 MB FP16 vs 382 MB "FP8"**), output `DataType`
> enumeration, and `Total Weights Memory` from the build log. *(Same family as the
> `jpeg_buf`-is-png all-zero FLOOR arm: a defect whose signature is indistinguishable from
> success.)*
>
> ⚠️ **Strongly-typed does not abolish fallback — it changes what fallback COSTS.** Builder-flag
> + no Q/DQ ⇒ the whole quantisation is lost and the accuracy gate **lies**; strongly-typed +
> Q/DQ ⇒ only the kernel/tactic falls back while the Q/DQ nodes still round the values, so the
> accuracy number stays **valid** and the loss is **latency**. Documented residual fallbacks:
> FP8 convs with kernel > 32 (SM89/90/120/121 — ⚠️ **SM110 not listed**, scope before quoting)
> and FP8 group convs.
>
> ⚠️ **Three caveats before building the gate on `IEngineInspector`:** (1) with dynamic shapes
> dims read `-1` and *"tensor format information will not be shown"*; (2) the layer-info JSON's
> `"Datatype"` is a **tensor dtype**, not provably the layer's **compute precision** — **MEASURE**
> that a usable field is emitted on our TRT 10.16.2 Thor build first; (3) ⛔ NVIDIA: *"The network
> layers will always report FP32. The engine layers are the ones which will have different
> precisions"* — never read `getPrecision()` on the **network**. **Polygraphy is not an
> attestation tool** (documented Heisenberg: `--trt-outputs mark all` *"can perturb the generated
> engine… which can hide the failure"*, breaking the Q/DQ fusions under test). ModelOpt's
> `print_quant_summary()` reports **inserted quantizers** (pre-build intent), not achieved
> precision.
>
> ⚠️ **#4590 itself re-verified OPEN 2026-08-30** (HTML + REST API agree): created 2025-10-04,
> updated 2026-01-09, **1 comment from `author_association: NONE`** — a user, not NVIDIA. No fix
> version, no workaround. ⚠️ A GitHub issue and a release note are **moving targets** — re-verify
> at gate-build time rather than inheriting this state.

## Relationship to the prior package

`…/2026-08-29-quant-gate-spec/RESULT.md` already designed the 4-stage fail-closed `quant_gate.py`
and already named strongly-typed + explicit Q/DQ as the structural fix. ⇒ **Nothing there is
retracted.** What this package adds: the #4590 reframe (F2), the reason a numerical stage cannot
catch it (F3), the accuracy-lie vs speed-loss distinction (F4), the JetPack/TRT-11 version wall
(F5), and three `IEngineInspector` caveats that would each have produced a wrong gate (F6).

## Decision asked

**Approve the migration to ModelOpt explicit Q/DQ + `--stronglyTyped` as the B1 build path**, and
**re-order `quant_gate.py` so the structural census runs BEFORE any accuracy stage.** Both are
implementation, no new hardware, no metered spend. Owner: DeployFlyWheel.
