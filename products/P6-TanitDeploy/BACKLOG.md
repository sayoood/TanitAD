# P6 TanitDeploy — BACKLOG

`Owner: TanitAD_DeployFlyWheel. Created 2026-08-23 from the content-verified
census of every deployment asset in the repo + on Thor. Ordered by
(value ÷ cost) × (does it unblock something else). Each item names its
BLOCKER and its DONE test — an item whose "done" is not checkable is not an item.`

**Legend** — P0 blocks the product · P1 next lever · P2 valuable, not urgent ·
🔴 durability risk · ⛔ blocked on a decision, not on work.

---

## P0 — the product cannot ship without these

### D-01 ⛔🔴 Ratify the MATERIALITY THRESHOLD for the four-family gate
**Why.** The best measured config (Thor: bf16 encoder + TRT-fp16 dyn-1..9 engine
+ batched fan, **60.30 ms p50**) has a paired 859-window gate whose verdict is
literally *"the optimised pipeline is NOT a drop-in for fp32 on this
checkpoint"* — `F_LONGITUDINAL` fired (`speed_bias` −0.0055 m/s
[−0.0076, −0.0034]; `accel_mae` +0.0228 [+0.0120, +0.0357]). The effect sizes are
**0.12–0.58 % of level**. With no pre-registered "how small is small enough",
that CI is unactionable in both directions.
**This is a PI/Master-Mind decision, not engineering.** A proposal of "1 % of the
fp32 level, pre-registered per family" exists and is unratified.
**BLOCKER:** decision. **DONE:** a threshold in `GOALS_AND_CLAIMS.md` with its
origin, and the existing gate JSON re-adjudicated against it — no new compute.
**Note:** ⛔ do not re-run the gate to "get a better number". The measurement is
sound; the *criterion* is missing.

### D-02 🔴 Rescue the deployment artifacts off single disks
**Why.** `champ30k` (133 MB, 4.1 h, the first collapse-free trunk) exists ONLY at
`thor:~/v7tiny/champ30k/`. Engines + ONNX (10.7 GB across `~/trt`, `~/trt_c2`,
`~/trt_c3`, `~/trt_d1`, `~/trt_deploy`) exist only on Thor and are gitignored.
The programme's worst measured failure mode is *good work stranded outside git*.
**Done in part 2026-08-23:** the rebuild recipes, descriptors and sha256s are now
in `engines/THOR_ENGINE_REGISTRY.md`.
**BLOCKER:** none. **DONE:** champ30k pushed to HF `Sayood/` (as the phase-0 arms
were), and every engine either rebuildable from a registry recipe or hashed.
**Cheap:** the 2.6 GB `~/trt` tree is *superseded* (random-init weights, batch-1
static) — it is reclaimable disk, not an asset.

### D-03 ⛔ Wire the optimised path into a real caller
**Why.** `propose_and_score(batch_fan=True)` appears only in tests and
harnesses. The only closed-loop driver is verified **heads-only** — no
imagination fan. Every fan optimisation makes the *designed* path affordable
without speeding up what actually runs. And a batch-9 engine on a serialised
caller is **slower** than batch-1, so shipping the engine without the caller is a
regression, not a win.
**BLOCKER:** needs an owner decision on which driver is canonical.
**DONE:** one driver calls the batched fan + `TRTPredictor`, and a test pins it.

---

## P1 — next levers, unblocked, cheap

### D-04a ⛔ Install `python3-dev` on Thor to unblock `torch.compile`
**Why.** All three compile arms died on `fatal error: Python.h: No such file` —
inductor builds a small CPython-API shim on the host and the headers are absent.
`apt install python3-dev` is one line and does **not** touch torch.
**BLOCKER:** ⛔ a system change on the fleet's ONLY GPU box, shared with other
agents' runs — needs an owner's ack, not a unilateral install.
**DONE:** headers present, then D-04 re-run.

### D-04 Test `torch.compile` on Thor — ATTEMPTED 2026-08-23, blocked
**Why.** ⭐ Triton **3.7.1 is present on Thor** (verified 2026-08-23) — the
"no Triton" constraint is a **dev-box** fact that has been silently generalised.
`torch.compile` has never been tried on the primary target. On the A40 it was the
single best rollout lever measured (52.89 ms vs graph 57.18).
**BLOCKER:** D-04a. **DONE:** compiled vs eager vs graphed, same harness, with
the deviation check; a compile failure is a result.
**Risk:** ⛔ compile is NOT bit-identical — it needs the §4.1 gate, unlike CUDA
graphs.

### D-16 ⛔ Gate the TF32 × CUDA-graph lever (22 %, and it is NOT free)
**Why.** MEASURED 2026-08-23: enabling TF32 *before graph capture* is **22.4 % /
22.2 %** faster than fp32+graph (within-process, ABAB-interleaved, both rounds),
taking the b1 tick from 12.519 → **6.10 ms (2.05×)**. But TF32 moves **every
learned tensor** — z_op **5.608e-02**, *larger than fp16 autocast's* 2.70e-02 —
while `plan.waypoints` reads exactly 0.0. ⭐ **The naive observable ranks TF32 and
fp16 in the OPPOSITE order to the one that matters.**
**BLOCKER:** D-01 (no materiality threshold) — same gate as fp16.
**DONE:** TF32×graph through the four-family gate; until then it may not be
described as free or bit-identical, and only **fp32 + CUDA graph** (1.59×,
genuinely bit-identical) is recommendable.

### D-05 Pin the kinematic integrator to fp32
**Why.** ⭐ MEASURED and the fix is **proven bit-identical**: `unicycle_rollout`
inherits the autocast dtype and accumulates 60 steps of distance in it (0.67 ULP
fp16 / 4.60 ULP bf16 at ~65 m). Upcasting the controls before the rollout
restores fp32 exactly at ~zero cost.
**BLOCKER:** touches `v6.py`/the emission head — needs the model owner's ack.
**DONE:** the upcast lands with its regression test in the same commit (a
deliberate-downcast arm that must FAIL the test).

### D-06 Re-register the precision screen against a TRAINED planner
**Why.** The 2026-08-23 screen ran on a stage-S-W checkpoint whose emission head
emits **exactly zero** controls — the fan is 8 identical straight lines, so the
screen measured integrator arithmetic, not the network. Its threshold was also
absolute (1e-2 m) against trajectories reaching 121 m.
**BLOCKER:** needs a checkpoint with planner losses on.
**DONE:** H-DEPLOY-4 re-registered with a **relative** bar on a trained planner.

### D-07 Build the encoder TRT engine at the deployed geometry (O2)
**Why.** Unblocked 2026-08-03 (`thor_c5_encoder_export.json`: exports clean, 1034
nodes, ORT rel-err 1.77e-05) and never built. The encoder is the larger stage
again (17.1 ms of a 60.3 ms tick at 256×256, ~30 ms at 176×624), and this is the
fallback if bf16 fails a decision gate. **BLOCKER:** none. **DONE:** engine +
descriptor + registry row + paired gate.

### D-08 Measure the tick at the DEPLOYED geometry (S4)
**Why.** Every end-to-end tick is at v1's 256×256. The 176×624 figure (~75 ms) is
`PROJ`, composed from per-stage measurements — and a composed number has never
survived contact with an end-to-end one in this programme.
**BLOCKER:** none (v5f intent engine already built). **DONE:** an end-to-end tick
JSON at 176×624 with its stage breakdown.

---

## P2 — valuable, not urgent

### D-09 Mixed-precision INT8 keeping the readout projection in fp16
**Why.** INT8 W+A lost on accuracy (**+0.021463 m ADE**, falsifier 0.02 fires)
*and* on latency (2.1 % slower than fp16 on the predictor) — but the failure is
**localised to one module**: `SpatialGridReadout.proj` collapses to cosine
**0.566** while every transformer block holds ≥0.9999. Excluding one projection
from quantization is the cheapest untested accuracy-per-byte lever.
⚠️ The INT8 evidence used a **train-cache proxy**, not the canonical val set — a
re-run must fix that regardless of the arm.

### D-10 bf16 decision-agreement (G5)
**Why.** The 6.76× encoder lever — the single biggest measured — rests on a
numerics rel-err, while the programme's own 4060 measurement **rejected** bf16 in
decision space (67.2 % agreement, 47.7 cm). ⭐ The 2026-08-23 v7-tiny profile
adds a **second independent signal**: bf16 deviates **30×** more than fp16 for
**zero** latency gain. Two measurements now point the same way. Either the lever
is confirmed in decision space or it must be withdrawn from the playbook.

### D-11 One deployment index across the two hubs
**Why.** The runbook's own evidence base, the ONNX/TRT export path, and the only
real INT8 benchmark all live under *Architecture & Inference*, not *Deployment &
Optimization*. A P6 reader will miss half the product's evidence.
**DONE:** one index in `products/P6-TanitDeploy/` pointing at every artifact,
wherever it lives.

### D-12 Repair the stale deployment docs
`FLAGSHIP_V1_INFERENCE_OPTIMIZATION.md` §8 lists Q1–Q5/Q9/Q10 as "in flight"
which `MODEL_REGISTRY.md:339-361` records **closed by measurement 2026-07-21**;
§2's "misses 10 Hz in every precision" is eager-only (registry: *meets 10 Hz at
p99 with 5.3× headroom*); §5.1's CEM-infeasibility is **self-retracted** and was
used to justify a pivot. `PRODUCTION_READINESS.md`'s blocker 2 closed 08-03.
`MODEL_REGISTRY.md:375-376` claims `combined_tick_harness.py` is not in HEAD — it
is tracked. **DONE:** each corrected in place with a dated line; registry wins.

### D-13 The `distance_keeping` family is structurally unmeasurable (G6)
`status: UNAVAILABLE`, n=0 — the ingest does not read `obstacle.offline`, which
is present in **97.44 %** of the corpus. One of the four binding families cannot
be computed. **Owner:** EvalFlyWheel; P6 is a consumer and must escalate, not
patch.

### D-14 FP8 / NVFP4 / 2:4 sparsity — zero measurements
Blocked behind D-01 (no gate criterion) and behind tooling maturity on Thor.
Do not start before D-09 answers whether mixed-precision fixes INT8.

### D-15 Reclaim ~7 GB of superseded engine trees on Thor
`~/trt` (2.6 GB, random-init weights, batch-1 static — provenance only) and
`~/trt_c2` (4.8 GB, opset-pair duplicates: `v1_pred_b1_fp16.onnx`,
`v1_pred_b1_fp32.onnx`, `v1_256x256_b1.onnx`, `fp_17_0/1.onnx` are **all md5
`8b6efc61…` — one byte-identical file under five names**). ⚠️ **The names assert
distinctions the bytes do not carry** (an "fp16 ONNX" and an "fp32 ONNX" that are
the same file — fp16 is a `trtexec` build flag, not an export property).
**DONE:** hashed, registry row, deleted with the PI's ack. Not urgent — Thor is
at 26 % of 937 GB.

---

## Done

| date | item |
|---|---|
| 2026-08-23 | **E-DEPLOY-1** — v7-tiny baseline profile on Thor: 9 arms, 5 controls, CUDA-graph grid, precision mechanism + proven fix. `2026-08-23-v7tiny-baseline-profile/` |
| 2026-08-23 | Engine registry with rebuild recipes + sha256 — `engines/THOR_ENGINE_REGISTRY.md` |
| 2026-08-23 | Content-verified census of every deployment asset (repo + Thor) |
| 2026-08-23 | **E-DEPLOY-2/3/4** — the Thor lever ladder: `torch.compile` (blocked on `python3-dev`), TF32 eager (3 % slower), TF32×graph (**+22 %**, paired A/B) and its trunk-deviation disqualification. `2026-08-23-thor-lever-ladder/` |
