# RESULT — the Thor lever ladder: `torch.compile`, TF32, and TF32 × CUDA-graph

`2026-08-23 · TanitAD_DeployFlyWheel · follow-on to
2026-08-23-v7tiny-baseline-profile, executed the same session on the same idle
Thor and the same champ30k checkpoint. Each hypothesis was written with both
outcomes before its run. Evidence class MEASURED (ours).`

## Headline

**TF32 under a CUDA graph is 22 % faster than fp32 under a CUDA graph — and it
is NOT free.** It perturbs every learned tensor in the stack, *more* than fp16
autocast does at the trunk (z_op 5.61e-02 vs fp16's 2.70e-02), while leaving the
plan output at exactly zero deviation. Anyone reading only the plan output would
ship it as "bit-identical".

| id | hypothesis | verdict |
|---|---|---|
| **H-DEPLOY-5** | `torch.compile` beats manual CUDA graphs on Thor | **REFUTED — but on a REPAIRABLE blocker, not a capability limit** |
| **H-DEPLOY-6** | TF32 is a net win on the eager tick | **REFUTED** — 3 % *slower* |
| **H-DEPLOY-7** | TF32 gives a real speedup under graph replay | **SUPPORTED** — 22.4 % / 22.2 %, both rounds |
| — | …and is it free? | ⛔ **NO.** Lossy at every learned tensor |

---

## 1. H-DEPLOY-5 — `torch.compile` on Thor: blocked, not beaten

`raw/thor_compile_FAILED_python_headers.json`

All three arms (`compile_default`, `compile_reduce-overhead`,
`compile_default_fp16`) failed **identically**:

```
/tmp/…/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
InductorError: CalledProcessError ['/usr/bin/gcc', …]
```

⚠️ **This is a missing `python3-dev` package, not a verdict on the lever.**
Inductor compiles a small CPython-API shim on the host and the headers are
absent. My pre-registration said "compilation fails ⇒ REFUTED", so REFUTED is the
verdict — but the *reason* changes what to do with it entirely, and reporting
only the verdict would be misleading.

⛔ **I did NOT install the package.** `apt install python3-dev` is a system change
on the fleet's only GPU box, shared with other agents' runs. It is one line, it
does not touch torch, and it is filed as backlog **D-04a** for an owner to
approve — not taken unilaterally.

**Two by-products from the compile logs, both useful:**
- `Not enough SMs to use max_autotune_gemm mode` — **inductor itself** judges
  Thor's 20 SMs too few for its autotune path. An independent corroboration of
  the saturation story behind H-DEPLOY-2.
- `TensorFloat32 tensor cores available but not enabled` — which is how TF32
  entered this session at all. **A failed experiment surfaced the next one.**

## 2. H-DEPLOY-6 — TF32 on the eager tick: REFUTED

`raw/thor_tf32_eager.json` · pre-registered bar: ≥ 5 % faster.

| arm | median ms | vs fp32 |
|---|---:|---:|
| fp32 (`highest`) | 12.369 | — |
| TF32 (`high`) | 12.734 | **0.97× — 3 % SLOWER** |

Consistent with H-DEPLOY-3: at batch 1 the tick is **launch-bound**, so making
matmuls faster changes nothing measurable. Defaults on Thor, recorded:
`matmul.allow_tf32 = False`, `cudnn.allow_tf32 = True`,
`float32_matmul_precision = "highest"`.

**Control** — the TF32 switch is live: a 2048² fp32 matmul changed by
**7.70e-02** when the flag flipped. Without that control, "no gain" would be
indistinguishable from "the flag never applied".

## 3. H-DEPLOY-7 — TF32 under a CUDA graph: SUPPORTED

`raw/thor_graph_tf32_paired_AB.json`

⚠️ **Why this needed its own run.** Two *different processes* suggested it:
fp32+graph 7.840 ms and tf32+graph 6.112 ms. But the nominal eager-fp32 arm read
**13.020 / 12.529 / 12.369 / 12.705 / 12.519 ms** across five processes today
(±4 %). A cross-process comparison is not admissible. So: one process, two graphs
captured from the same loaded model differing only in the TF32 flag **at capture
time**, replays **ABAB-interleaved**, two rounds.

| round | fp32-graph | TF32-graph | gain |
|---|---:|---:|---:|
| 1 | 7.864 ms | 6.103 ms | **22.4 %** |
| 2 | 7.867 ms | 6.121 ms | **22.2 %** |

Both rounds agree to 0.2 pp. **The full lever stack: eager fp32 12.519 ms →
TF32 + graph 6.10 ms = 2.05×.**

The gain appears only under graphs because that is when it can: graphs remove the
fixed ~7.45 ms launch cost, after which compute is a large enough share of the
tick for a faster matmul to show.

### ⛔ RETRACTED IN-FLIGHT — a clobbered output tensor read as a numerical result

The first run of this probe reported `bit-identity: fp32graph 5.613e+01` — a
"56 m deviation" that is **exactly `waypoints_absmax`**. Cause: a CUDA graph's
static output tensor is valid only until its memory pool is reused, and I read
graph A's output *after* capturing graph B. The tensor had been clobbered, not
perturbed. Fixed by replaying each graph and reading its output immediately, and
a permanent control now **voids the run** whenever a deviation equals the
reference's own magnitude. Root-cause class: *reading a buffer whose lifetime had
ended* — a lifetime error wearing a numerical result's clothes.

## 4. ⛔ The headline that did not survive: TF32 is NOT free

`raw/thor_tf32_trunk_deviation.json`

The paired A/B showed TF32+graph deviating **0.0** from strict fp32 on
`plan.waypoints`. That invites "a free 2×, bit-identical". It is wrong, and the
baseline package already said why: on this stage-S-W checkpoint the waypoints are
`unicycle_rollout(0, 0, v₀)` and **cannot** respond to matmul precision. So the
learned tensors were measured directly:

| tensor | TF32 max abs dev | scale | relative | moved? |
|---|---:|---:|---:|---|
| `z_op` (trunk latent) | **5.608e-02** | 8.572 | 6.5e-03 | ✅ **yes** |
| `z_tac` | 1.148e-02 | 2.608 | 4.4e-03 | ✅ yes |
| `z_str` | 7.079e-03 | 2.727 | 2.6e-03 | ✅ yes |
| `plan.feat` | 2.515e-03 | 1.879 | 1.3e-03 | ✅ yes |
| `plan.waypoints` | **0.000e+00** | 56.13 | 0 | ❌ **no** |

Control: strict fp32 repeated twice is bit-identical at **every** stage, so these
deltas are dtype, not noise.

⭐ **TF32 perturbs the trunk MORE than fp16 autocast does** — z_op 5.61e-02 vs
fp16's 2.70e-02 — while presenting a *cleaner* plan output than fp16. The
observable that a naive screen would use ranks the two levers in the **opposite
order** to the observable that matters.

⇒ **TF32 ships only behind the four-family gate, exactly like fp16.** It may not
be called free or bit-identical.

⚠️ **This retroactively narrows a claim in the baseline package.** There,
CUDA-graph replay was called bit-identical — that claim **stands**, because graph
replay changes no arithmetic at all and was verified against its own eager arm at
every batch and dtype. What must not be done is to inherit that property for
**TF32**+graph, which is a different lever that happens to share the mechanism.

## 5. The playbook after today

| rung | effect at b1 | accuracy status |
|---|---|---|
| **CUDA graph** (fp32) | 12.5 → 7.86 ms, **1.59×** | ✅ **bit-identical — free, no gate needed** |
| **+ TF32 at capture** | 7.86 → 6.11 ms, **+22 %** (2.05× total) | ⛔ **lossy — needs the four-family gate** |
| fp16 autocast, b1 | **slower** (1.21×) | ⛔ lossy, and unmeasured on a trained planner |
| fp16 autocast, b≥4 | 1.26–1.39× | ⛔ same |
| `torch.compile` | unknown | ⛔ blocked on `python3-dev` (D-04a) |

**The single safe recommendation today remains CUDA graphs at fp32.** Everything
faster is lossy and queues behind the gate that backlog D-01 is blocked on.

## 6. Deliverable manifest

| artifact | location |
|---|---|
| Compile failure raw | `repo:products/P6-TanitDeploy/2026-08-23-thor-lever-ladder/raw/thor_compile_FAILED_python_headers.json` |
| TF32 eager raw | `repo:…/raw/thor_tf32_eager.json` |
| Paired A/B raw (post-fix) | `repo:…/raw/thor_graph_tf32_paired_AB.json` |
| TF32 trunk deviation raw | `repo:…/raw/thor_tf32_trunk_deviation.json` |
| 4 probe scripts | `repo:…/code/` (also `thor:~/`) |

Nothing here exists in only one place. **Nothing was installed on Thor**; no
`uv pip install` ran; Thor was verified idle before each job.
