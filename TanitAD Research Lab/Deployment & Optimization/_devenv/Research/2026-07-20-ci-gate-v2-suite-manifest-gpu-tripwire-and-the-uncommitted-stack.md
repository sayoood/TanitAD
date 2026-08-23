# 2026-07-20 — ci_gate v2 (suite manifest + CUDA tripwire), the uncommitted-`stack/` strand, and the July sim/tooling sweep

Agent: Tools & DevEnv (Monday, W4). Branch `agent/tools-devenv-20260720`, worktree
`C:/Users/Admin/wt-tools-0720` off `c4d8451` (`agent/tools-devenv-20260718`).
Resource: local RTX 4060 (CUDA experiments) + dev-box CPU + web sweep. Wall ~2.3 h. $0.

Backlog executed: **P0 #2 — "ci_gate extension: fold the newly-merged suites into the
one-command gate; report the new green total + wall-clock; keep it under 5 min or shard it."**
Two measured experiments (G-I / D-029 ≥2): the gate extension itself, and a CUDA
device-parity tripwire run on the 4060.

---

## 1. The measured finding that reframed the task: the test suite has ZERO GPU coverage

Before extending the gate I inventoried what it actually guards. `grep -rl cuda
stack/tests/` returns **nothing** — all 396 (worktree) / 531 (Drive tree) tests are
CPU-only. Meanwhile every trainer, every eval and every deploy tick runs on a GPU:
pod2 flagship, pod1/pod3 reference arms, the A40 eval pod, the Orin export path, and
Prod-Opt's CUDA-graph work (`b984e04`, 11.16 ms deploy tick) — plus Architecture's
`kamm_circle` `0*inf` NaN-gradient trap (`2f1dae6`) and the MooseFS mmap bus-error
fix that only manifests across DataLoader worker boundaries (`986b688`).

So the class of breakage that has actually cost this program time — device/dtype
placement, a batch-statistic layer that behaves differently on device, a NaN the CUDA
kernel produces and the CPU path does not — was **structurally invisible to CI**. A
gate can only ever be as good as the surface it observes; extending the gate's
*bookkeeping* while leaving that surface untouched would have been the wrong increment.

Verdict: the extension ships in two halves — the suite manifest the backlog asked for,
**and** a CUDA half.

## 2. Experiment A — `tools/gpu_tripwire.py`, CUDA device parity on the 4060

Four probes against the real `WorldModel(smoke_config())`, identical seeded inputs
CPU vs CUDA. Hardware: RTX 4060 8 GB, torch 2.11.0+cu128, fp32. Wall **1.7 s**, $0.

| Probe | Asserts | Measured |
|---|---|---|
| `P1_encode_parity` | `encode` CPU vs CUDA | **9.537e-07** (tol 1e-3) PASS |
| `P2_imagine_parity` | operative predictor, all horizons | **7.153e-07** @ h=1 (tol 1e-3) PASS |
| `P3_i2_on_device` | I2 batch-1 vs batch-8, **on CUDA** | **1.659e-07** (tol 1e-4) PASS |
| `P4_backward_finite` | one `loss.backward()` on CUDA | 0 non-finite grad tensors PASS |
| I8 proxy | batch-1 `encode` latency | **0.85–1.43 ms** |

**Result vs expectation:** expected fp32 CPU-vs-CUDA deviation ~1e-5; measured ~1e-6,
i.e. an order of magnitude tighter. The default tolerance is therefore set at **1e-3
(~1000× headroom)** — loose enough not to flake on a driver/kernel change, tight enough
that a real device bug is orders of magnitude past it. P3 keeps the stricter 1e-4 the
I2 doctrine already mandates.

**Falsifier (pre-registered, executed):** "if the probes cannot fail, they are not
comparing anything." `test_an_impossible_tolerance_fails_the_parity_probes` runs the
same probes at `tol=0.0` and asserts P1+P2 both fail. They do. The probes are live,
not vacuous.

**Honest caveat (readiness = validated, not production):** this is a *parity* tripwire
on the smoke-scale model, not a numerical-correctness proof of the training path. It
does not exercise bf16/AMP, multi-GPU, the CUDA-graph capture path, or the mmap
dataloader. Gap to production: add a bf16 arm and a CUDA-graph capture/replay probe —
both are cheap and are now backlog P0.3.

The batch-1 latency reading (0.85–1.43 ms, varying with concurrent load) is the cheap
Orin proxy for I8; it is *informational only* and never a gate reason, because a shared
dev box cannot produce a stable latency number — see §4.

## 3. Experiment B — `tools/ci_gate.py` v2: the suite manifest, and the sharding question answered

### 3.1 The gap a node tripwire cannot close

v1 guarded named nodes (`--require test_i2_batch_consistency_of_encoder`). A named node
only guards a node somebody thought to name. The way coverage actually disappears in
this repo — six agents editing one tree — is a whole module being deleted, renamed, or
quietly halved, which v1 would report as a cheerful green.

v2 adds `SUITE_MANIFEST`: 16 load-bearing modules pinned to a **collected-count floor**
(instrument doctrine, the calib pair, the three reference arms, flagship, gates,
metrics/eval-behavior, the data contracts, resim/scena). Plus `--min-total` (default
390) for wholesale loss, e.g. a broken `conftest.py` deselecting half the tree. Adding
tests is always fine; removing them has to edit the dict on purpose.

Deliberately *not* all 46 modules — a hand-maintained 46-entry manifest rots, and a
rotted manifest is worse than none. Targeted list + global floor covers both failure
shapes at a maintenance cost that survives.

**Backlog correction (G-A honesty):** the backlog item named "test_eval_behavior (22)",
"the metric-suite tests (22)" and a "calib trio (test_calib + test_calib_r1 +
test_physicalai_rig)". Measured reality on `c4d8451`: `test_eval_behavior` = **13**,
`test_metrics` = **22**, `test_scena` = **22**, and **`test_calib_r1.py` does not exist**
— it was folded into `test_calib.py` (12) at some point, so the "trio" is a pair
(`test_calib` 12 + `test_physicalai_rig` 5). The manifest encodes what is true, not
what the backlog remembered.

### 3.2 The skip-semantics bug the live run found

First full run **failed the gate** on `tests.test_scena::test_minilm_search_ranks_sc01
= skipped` — my first cut treated any non-`passed` status as suite rot. Wrong: a skip
for an optional model download is legitimate, and gating on it makes the manifest
unusable within a week. Fixed to: `failed`/`error` fail; skips are fine — **unless the
whole module is skipped**, which is coverage loss wearing a green hat and is exactly
what the manifest exists to catch. Two falsifiers pin both halves.

### 3.3 Measured green total, wall-clock, and the sharding verdict

Clean run, nothing else on the box, `--gpu-smoke require`:

| Tree | Collected | Result | Suite wall | + GPU | Tall pole |
|---|---|---|---|---|---|
| worktree `c4d8451` (off-Drive, C:) | **396** | 394 passed / 2 skipped | **39.0 s** | 1.7 s | `test_replay…regression_gate` **8.02 s** |
| shared Drive tree (`fcbab02` + uncommitted) | **531** | 529 passed / 2 skipped | **60.2 s** clean, 65.2 s under load | 1.8 s | 7.91 / 8.89 s |

Both trees **GATE PASS** with the full v2 manifest + `--min-total 390` + the CUDA
tripwire, which also validates that the 16 manifest floors hold across a tree with 135
extra tests in it (§4) — the manifest is a floor, not a fingerprint.

**Sharding verdict: NOT NEEDED.** The backlog's condition was "keep it under 5 min or
shard it". The largest tree measured **60.2 s — 5× under the ceiling**. Building a
sharder now would be speculative complexity; the trigger to revisit is a wall over
~150 s (the new `--max-wall-seconds` default, ~2.5× today's worst).

**Budgets set from measurement:** per-test 15 s (tall pole 7.9–8.0 s clean), wall 150 s.

### 3.4 A caveat worth recording: these timings are contention-sensitive

An earlier run of the same suite, executed while a second pytest process was running,
reported **65.0 s wall and a 14.90 s tall pole** — nearly double, and within 0.1 s of
tripping the 15 s per-test budget. So: **the gate's slow-test budget can false-positive
if an agent runs it concurrently with other work.** Two consequences — (a) the 15 s
budget stays where it is rather than tightening toward the original 6 s intent, and
(b) backlog P0.2 ("speed up `test_replay`, 10.86 s") was **partly an I/O/contention
artifact, not a fixture cost**: the same test measures 7.2–8.0 s clean and 14.9 s
contended. Re-scoped, not retired.

### 3.5 Falsifier suite

`tools/tests/` = **57 tests, 15.5 s** (was 15 for `session_guard` alone). New v2
falsifiers: suite at/below floor, module renamed away, suite red, entirely-skipped
suite, legitimate skip stays green, `--min-total`, spec-parser forms + bad floor,
GPU probe failure is a reason, GPU absent under `require` is a reason, `--gpu-smoke
warn` never blocks, `--json` payload shape, manifest-normalization non-vacuity.

## 4. The bigger strand: 40 uncommitted `stack/` paths on the shared tree

Running v2 against both trees produced a **135-test discrepancy** (396 vs 531) that had
no branch explanation — the worktree's commit contains the Drive tree's HEAD. Root
cause, via `git status --porcelain -uall`:

> The shared Drive working tree carries **40 uncommitted `stack/` paths — 22 of them
> UNTRACKED**: 12 whole test modules (~135 tests: `test_lake_*` ×4, `test_refb_labels_v2`,
> `test_speed_input`, `test_vision_levers`, `test_gated_intent`, `test_anchor_tactical`,
> `test_invdyn_gradscale`, `test_labels_v2_wiring`, `test_eval_speed_ckpt`), 9 untracked
> `tanitad/lake/*` + `eval/ckpt_compat.py` + `train/decorr.py` modules, and 18 modified
> core files (`config.py`, `fourbrain.py`, `predictor.py`, `refa.py`,
> `flagship_losses.py`, `lake/schema.py`, + 10 scripts). **None of it exists in any
> commit, on any branch, anywhere.**

This is a strictly worse strand class than D-026's unmerged branches: an unmerged
branch is at least *committed and pushed*. `session_guard` v1 reported this tree as
having a clean source state, because it only ever looked at hub prefixes.

**Fix shipped:** `session_guard` gains a source check over `stack/` + `tools/`, listing
untracked separately from modified (an untracked file has no other copy). WARN by
default — a mid-work tree is legitimately dirty — BLOCK under `--strict`.

**Second bug, caught by its own falsifier:** `git status --porcelain` collapses a wholly
untracked directory into a single `?? stack/` row. A guard whose job is to *name* the 12
missing modules must not do that → switched to `--untracked-files=all`. The test
`test_source_rows_survive_quoted_paths_with_spaces` failed first, which is how the bug
surfaced before shipping rather than after.

### 4.1 Live escalation list (for the orchestrator's D-026 sweep, 2026-07-20)

`python tools/session_guard.py --repo <Drive tree>` — debt has **grown** since 2026-07-18:

| Class | 2026-07-18 | **2026-07-20** |
|---|---|---|
| uncommitted hub deliverables | 5 | **30** |
| uncommitted `stack/` paths | (not checked) | **40** (22 untracked) |
| unmerged `agent/*` branches | 9 | **11** |
| stale INTAKE verdicts (>3 d) | 5 | **8** |

Highest-value items in the uncommitted-hub set (each is a full deliverable that exists
only on one Drive volume): the entire Benchmarks & Eval intake package
`2026-07-19-alpasim-closedloop-v1/` **including its results JSONs**, Architecture's
`V3_HIERARCHICAL_PLANNING_DESIGN.md` + `V3_GOAL_VOCABULARY_V1.md` + four 07-19 research
notes, Data-Eng's `TANITDATASET_V1_STRATEGY.md` + three surveys, `HYPOTHESIS_LEDGER.md`,
`DECISIONS.md`, `PROJECT_STATE.md`, and the `2026-W33` progress report.

Newly stale INTAKEs since last run: `2026-07-15-h15-logging-fidelity`,
`2026-07-15-baseline-floor`, `2026-07-15-pandaset-loader` (5 d each). Still unfilled at
11 d: `lal-v2-anticipation`, `physicalai-r1-selection`, `models-predictor-failfast`,
and this discipline's own `testsuite-io-profiling`.

## 5. Literature & tooling sweep (2026-07-10 → 2026-07-20)

Method: arXiv export API over cs.RO (driving / closed-loop / world-model / simulator
filter, 2026-07-03 → 2026-07-17, 38 entries by title) + 6 targeted web searches + 7
direct fetches. Delegated fan-out, findings verified against the linked sources.

1. **Rerun 0.34.0 / 0.34.1 (2026-07-06 / 07) — Viewer MCP.** The viewer now exposes an
   MCP server: an agent can *see and interact with what the viewer renders*. Also
   `VoxelGridMap`, transform-debugging UI, live-stream stack-overflow fix (0.34.1).
   Breaking API changes → migration guide required.
   [releases](https://github.com/rerun-io/rerun/releases)
   **Why it matters:** this is the highest-leverage item in the sweep for a
   1-person + agents team. Every rollout-overlay claim an agent makes today is
   unverified assertion; Viewer MCP lets the agent look at its own render and
   self-correct — directly serving the TanitEval viz standard (camera projection +
   metric BEV inset + decoded maneuver overlay).
   **Verdict: GO**, on a branch. Est. 1–2 h SDK bump + `corpus_overlay.py` migration,
   +~30 min to wire the MCP into an agent tool list. Pin **0.34.1**, not 0.34.0.
   → new backlog **P0.1** (it also *is* the long-parked "episode → .rrd" item, upgraded).
2. **"Validate the Dream Before You Trust Its Verdict" (arXiv 2607.07196, 2026-07-08).**
   RSS-2026 workshop: a world model used as a *test oracle* must be accredited first;
   proposes an L0–L4 admissibility ladder from VV&A / SOTIF practice. Key result: **the
   model that ranks higher on visual generation quality ranks lower on action-following.**
   [abs](https://arxiv.org/abs/2607.07196)
   **Why it matters:** the cleanest external statement of the trap our own numbers hit
   (open-loop ADE 0.45 m → closed-loop 1.69 m). Gives the paper a standards-anchored
   vocabulary instead of a home-grown gate taxonomy, and argues *against* spending
   compute on making imagination look prettier. **Verdict: GO (read + cite, 45 min, no
   compute).** Hand-off: Benchmarks & Eval (TanitEval tier definitions).
3. **TerraZero (arXiv 2607.13028, 2026-07-14, Applied Intuition).** Procedural driving
   sim, C engine, **1.3 M agent-steps/s on one GPU**, pure-RL policies, tops InterPlan.
   [abs](https://arxiv.org/abs/2607.13028)
   Exactly the shape of closed-loop harness our envelope can afford (no rendering).
   **Verdict: NO-GO now, WATCH** — no code released, commercial vendor; assume closed.
   $0 today; re-check in 2–4 weeks. If code lands it is a first-priority integration.
4. **DynaDreamer (arXiv 2607.13410, 2026-07-15).** A physics-informed ego-dynamics
   encoder compresses ego-state history into a context that *modulates* a causal-
   Transformer world model, with a dynamics predictor keeping it synchronized during
   rollout; +28 % urban / +61 % highway, +73 % on an unseen chassis with no retraining.
   [abs](https://arxiv.org/abs/2607.13410)
   **This is the published relative of our own speed/scale reset** (v0 as a 3rd action
   channel: REF-A 3.73 → 0.83 m fwd-ADE, speed-R² 0.61 → 0.965) and of the guarded
   yaw-rate conditioning on `35956b2`. Their rollout-time dynamics-propagation is a
   principled generalization of our channel concat and targets the longitudinal lever
   (83 % of 2 s error along-track). **Verdict: GO as design input, not a dependency** —
   no code URL, no stated scale. Hand-off: Architecture & Inference (H4/H25).
5. **Orbis 2 (arXiv 2607.15898, 2026-07-17, Freiburg).** Hierarchical driving world
   model (coarse predictor + detail generator), **diffusion-forcing pretraining →
   teacher-forcing fine-tuning**; code + checkpoints advertised.
   [abs](https://arxiv.org/abs/2607.15898)
   Two transfers despite the scale gap: the hierarchy validated *inside* the world model
   (our V3 bet, but one layer down), and a reusable training-schedule recipe for rollout
   stability — which is the mechanism behind our closed-loop divergence, and costs only
   a schedule change. **Verdict: PARTIAL GO — read the training loop (30–60 min); do NOT
   run their checkpoints** (generative video WM, will not fit the 4060 and would burn
   A40 time the arms need).
6. **"Think at 5 Hz, Act at 20 Hz" (arXiv 2607.15621, 2026-07-17).** Async fast/slow
   VLA: slow branch ~5 Hz, fast branch emits control every tick, coordinated by a
   per-layer KV cache. [abs](https://arxiv.org/abs/2607.15621)
   Rate-decoupling is the deployment answer to "our imagination step will not run at
   control rate on an Orin". **Verdict: WATCH, low priority** (30 min read; only
   actionable once a closed-loop planner exists). Filed to the edge-export seam.
7. **Correction to our own KB:** we recorded JetPack **7.1** as the Orin export target.
   **JetPack 7.2 (Jetson Linux 39.2) shipped 2026-06-02** and brings the **Orin family
   into the JetPack 7 line** (CUDA 13.2.1, TensorRT 10.16.2, unified installer for Orin
   and Thor). The NVFP4-is-Thor-only / Orin-targets-FP8-INT8 story is unchanged, but the
   export path should be planned against **7.2, not 7.1**.
   [JetPack](https://developer.nvidia.com/embedded/jetpack) ·
   [7.2 announcement](https://forums.developer.nvidia.com/t/jetpack-7-2-jetson-software-goes-agentic-with-jetson-linux-39-2/372056)

**NOTHING NEW ON** (queries run, only already-known material returned): NVIDIA
AlpaSim/AlpaGym/Alpamayo since 07-10 (34 B Super still unshipped); CARLA releases
(0.10.0 still current; the RunPod `NVIDIA_DRIVER_CAPABILITIES` blocker unchanged);
Bench2Drive/-Speed; **dev tooling — experiment tracking, dataset streaming, CI-for-ML,
pytest speedups, RunPod/Colab burst (MLflow/W&B/LeRobot/litdata/energon): zero July-2026
announcements retrieved.** Coverage gap to close next run: a dedicated cs.CV listing
pass (this sweep ran cs.RO + cross-lists only).

**Hand-off to Benchmarks & Eval (D-028 seam — benchmark/dataset releases are theirs):**
nuTruck (2607.13704), M⁴World (2607.14005), CARLA-GS (2607.07601), OmniSCS (2607.09764),
CLEAR (2607.02841).

## 6. Backlog item retired by another agent (cross-agent hygiene)

My **P1.0 "AlpaSim single-A40 eval-harness smoke test"** was executed on 2026-07-19 by
the investigation agent and is **answered NO-GO**: the eval pod is itself an
unprivileged container with **no nested container runtime**, and AlpaSim's NuRec
renderer ships only as `nvcr.io/nvidia/nre/nre-ga:26.04` with no source form. Policy
side GO, renderer NO-GO, ~1.5 GB/scene and <2 GB VRAM would fit a proper host. See
`Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/INTAKE.md`
(itself uncommitted — §4). **Retired from my backlog**; the residual ask is infra
(a docker-capable GPU host), which is a Sayed decision, not a tooling task.

## 7. Actionable recommendations (G-B)

1. **Orchestrator, this week:** run `python tools/session_guard.py --repo <Drive tree>`
   and land the §4.1 list. The uncommitted `stack/` set (40 paths, 22 untracked) is the
   priority — it is one `git clean` away from being gone. Then the 8 stale INTAKEs.
2. **Every agent, now:** `python tools/ci_gate.py --rootdir stack --gpu-smoke require`
   before push, on any box with a GPU. On a CPU-only box use `--gpu-smoke warn` — it
   prints and never blocks.
3. **Architecture & Inference:** DynaDreamer (§5.4) — the rollout-time ego-dynamics
   propagation is the principled form of our v0 channel and targets the longitudinal
   83 %. Measure-first at the next v2 milestone. Orbis 2's diffusion-forcing →
   teacher-forcing schedule (§5.5) is a cheap rollout-stability lever.
4. **Benchmarks & Eval:** adopt the L0–L4 admissibility ladder (§5.2) as the TanitEval
   tier vocabulary; it is the citable form of our open-loop ⊥ closed-loop finding. Plus
   the five-paper dataset/benchmark hand-off in §5.
5. **Edge seam (Prod-Opt / Architecture):** re-target the ONNX→TensorRT plan at
   **JetPack 7.2**, not 7.1 (§5.7).

## 8. Hypothesis-ledger impact

None. This run's outputs are process/instrument, not evidence about H0–H26 — recorded
here so the absence is deliberate rather than an omission (G-D).

## 9. Readiness statement (D-029)

- `tools/ci_gate.py` v2 — **validated** (57 falsifiers, green on two real trees, two
  bugs caught pre-ship). Gap to production: no CI *runner* executes it automatically;
  it is still a discipline an agent must perform. → P0.2.
- `tools/gpu_tripwire.py` — **validated** (4/4 probes on real hardware, falsifier-proven
  non-vacuous). Gap to production: no bf16/AMP arm, no CUDA-graph capture probe, smoke-
  scale model only. → P0.3.
- `tools/session_guard.py` source check — **validated** (6 new falsifiers, live run
  found a 40-path strand the prior version reported as clean). Gap: still advisory for
  source; only `--strict` blocks.
