# Tools & DevEnv — 2026-07-18 (W4, Monday)

Author: tools-devenv-agent · branch `agent/tools-devenv-20260718` (worktree
`C:/Users/Admin/wt-tools-0718`, off HEAD `fcbab02`).

**TL;DR.** Shipped the top fleet-directive item: `tools/session_guard.py` — the D-026
stranded-work guard every agent now runs at session end (protocol-wired into G-F/G-I).
Measured it against the **live** repo tree: it catches the exact debt class the
2026-07-17 fleet review had to clean by hand — **5 uncommitted hub deliverables
(BLOCK), 9 unmerged `agent/*` branches vs tip, 5 stale INTAKE verdicts (9d/5d old)**.
Literature: **AlpaSim + AlpaGym are now publicly clonable (Apache-2.0)** — AlpaSim is a
viable single-A40 eval-harness smoke-test; AlpaGym is 2-GPU-gated (reference only).
TensorRT **Edge-LLM** (JetPack 7.1) is the current ViT/VLM edge-export path but **NVFP4
is Thor-only → Orin must target FP8/INT8**.

---

## 1. Implementation increment (backlog P0.1 / fleet directive #1) — the D-026 guard

**Problem.** The fleet review merged ~15k stranded lines across 8 branches, plus
uncommitted hub deliverables and INTAKE packages sitting for days with no verdict.
That debt is recurring and mechanical, so it should be a gate, not a manual audit.

**Built** `tools/session_guard.py` (+ `session_guard.ps1` Windows wrapper, `tools/README.md`,
`tools/tests/test_session_guard.py`). Stdlib-only, ASCII-clean stdout, OS-agnostic.
Three checks, exit-code-encoded:

| Check | Detects | Severity |
|---|---|---|
| (b) uncommitted **hub deliverables** | modified/untracked files under `TanitAD Research Hub/`, `Project Steering/`, `PROJECT_STATE.md`, `DECISIONS.md` | **BLOCK** (exit 1) |
| (a) **unmerged `agent/*` branches** vs tip | `git rev-list --count <tip>..<branch> > 0`; current session branch is info-only, never blocks | WARN (block under `--strict`) |
| (c) **stale INTAKE verdicts** | `incoming/<date>-slug/INTAKE.md` with an unfilled `ORCHESTRATOR VERDICT` older than `--max-intake-age-days` (default 3), by the folder date prefix | WARN (block under `--strict`) |

Design notes worth keeping:
- **Tip = `HEAD` by default**, not `origin/main` — `origin/main` (`0f93b98`) is
  intentionally diverged in this repo; the working integration tip is the checked-out
  HEAD. `--base` overrides.
- **A porcelain-parse bug was caught by its own falsifier test** before it shipped:
  the shared `git()` helper did `.strip()`, which removes the leading status column
  space from the *first* `git status --porcelain` line (` M path` → `M path`), shifting
  the fixed-offset `[3:]` path parse. Fixed to `.rstrip()` (preserve leading, drop
  trailing). This is exactly the class of silent tooling bug the falsifier suite exists
  to catch — an untested guard would have missed the first modified hub file every run.
- **Verdict-unfilled hardening:** treats the template menu (`integrate / … / reject`),
  empty, *and* non-committal tokens (`_pending_`, `tbd`, `todo`, `none`, `n/a`) as
  "no decision"; a package with **no** `## ORCHESTRATOR VERDICT` heading at all is
  unfilled (found live: `cosmos-robustness-first-pass` has a `_pending_` verdict only
  in the author section — correctly flagged).

### Measured result (the experiment — real compute: local box, git over the live tree)

Falsifier suite: **15/15 pass, 5.2 s** (`venvs/tanitad` py3.13, RTX-4060 dev box). Each
test builds a throwaway git repo and drives the real guard end-to-end.

Run against the **live Drive working tree** (`fcbab02`, the actual current debt) — this
is the falsifier that matters, "does it catch real stranded work":

```
[BLOCK] 5 uncommitted hub deliverable file(s)   -> PROJECT_STATE.md, HYPOTHESIS_LEDGER.md,
        External Anaysis.md, refa-frozen-encoder-improvement-plan.md, external-survey-derivation.md
[WARN]  9 unmerged agent branch(es) vs tip      -> phase0-supervised-hardening(+7),
        phase0-highway-dataset(+3), data-engineering-20260711(+2), …, prod-opt-20260718(+1)
[WARN]  5 INTAKE package(s), unfilled verdict    -> lal-v2-anticipation(9d), physicalai-r1-selection(9d),
        models-predictor-failfast(9d), testsuite-io-profiling(9d), cosmos-robustness-first-pass(5d)
RESULT: BLOCK   (exit 1)
```

**Falsifier verdict — PASS.** Expectation: the guard surfaces the review's debt class
with zero manual reading. Observed: all three classes flagged, true-positives
spot-verified (e.g. `testsuite-io-profiling/INTAKE.md` verdict is still the template
menu; `cosmos` has no verdict heading). No false positives on non-hub paths (verified
by test + the guard leaving `tools/` untracked files unflagged when run on my worktree).

**Escalation surfaced by the guard (for the orchestrator, D-026 sweep):** 5 INTAKE
packages have sat un-triaged for 5–9 days — `2026-07-09-testsuite-io-profiling` is
this discipline's own (KB says "shipped via intake" but the verdict was never written
back). The tool now makes that gap impossible to miss at session end.

**Readiness: VALIDATED.** It runs, is tested (15 falsifiers), and produced a correct
result on live data. Gap to *production*: wire it into the actual session-end automation
(currently a manual protocol step + a hook opportunity), and let the orchestrator act on
the WARN list (auto-merge is deliberately *not* done by the guard — flagging is safe,
force-merging an agent branch from an agent session is not).

**Protocol wiring.** `agents/_common-protocol.md` G-F now mandates a `session_guard`
run before session end (BLOCK on uncommitted hub deliverables → PASS required); G-I
references the WARN list as the mechanical stranded-branch check. This is the
highest-leverage part: the tool only prevents debt if every agent runs it.

## 2. Literature sweep (delta since 2026-07-17)

Deepest first. Full agent sub-report archived in this run's transcript.

1. **[headline] AlpaSim is public & clonable — Apache-2.0.** `NVlabs/alpasim`
   (tag v2026.5). Microservice closed-loop AV sim (renderer/physics/runtime/controller/
   driver/traffic over gRPC), NuRec default neural renderer, links out to ready policies
   (Alpamayo-R1/1.5, VaVAM, Transfuser); eval data = `PhysicalAI-AV-NuRec` on HF (900+
   reconstructed scenes). **Go/no-go for us:** NuRec/OmniDreams rendering is GPU-heavy +
   multi-service → not a 4060 job; realistic path is **one A40 pod (renderer+runtime)**,
   or keep CARLA for cheap iteration. Worth a single-A40 smoke test as an *eval harness*
   for our world-model policy. https://github.com/NVlabs/alpasim
2. **[go/no-go] AlpaGym (closed-loop RL) public but 2-GPU-minimum.** `NVlabs/alpagym`,
   Apache-2.0, GRPO on AlpaSim+Cosmos-RL. Docs: the default 10B policy **requires two
   GPUs** (`alpamayo_1_5_local_2gpu_smoke`). **No-go on a single 4060/A40** for the stock
   config; viable only with a much smaller policy (ours, <100 M) or a 2×A40 pod. Confirms
   the standing verdict: reference architecture for our Phase-1 self-play story, not a
   drop-in. https://github.com/NVlabs/alpagym
3. **[reference asset] Alpamayo-R1-10B weights live on HF (~22 GB).** Chain-of-thought VLA
   on a Cosmos-Reason backbone; a usable open teacher/reference policy for distillation
   and for driving AlpaSim rollouts. **Alpamayo 2 Super (34B) has NOT shipped** ("summer
   2026" still). 34B stays a hard no for our envelope regardless. https://huggingface.co/nvidia/Alpamayo-R1-10B
4. **[deployment] TensorRT Edge-LLM (JetPack 7.1) = the ViT/VLM edge-export path, but
   chip-gated.** HF→quantize→ONNX→engine, `tensorrt-edgellm-export` handles visual
   encoders (`--visual_quantization fp8` for ViT). **NVFP4 (the 4× memory win) needs
   Thor/SM110+ (Blackwell) at runtime — Orin cannot run NVFP4, only FP8/INT4.** Deployment
   rule: **lock the target chip first** — Orin → FP8/INT8; Thor → NVFP4. (Hand-off to
   Architecture/Prod-Opt: keep encoder input static `[6,256,256]` for a static-shape
   ONNX→TRT-FP16 engine.)
5. **[gotcha, reconfirmed] ViT INT8 on Orin can regress 2.7×.** NVIDIA dev-forum: TRT
   Model-Optimizer INT8 on a ViT-S+DPT stack ran 2.7× *slower* than FP16 on Orin Nano
   (ViT layers → non-optimal INT8 kernels). **Don't assume INT8 = speedup for our
   encoder; benchmark FP16 vs INT8 per-layer before any edge quantization.**

**Discarded/watch-list (not actionable this cycle):** CARLA 0.9.16 (Sept 2025, has NuRec
+ Vulkan headless + USD SimReady) — solid A40 iteration option but not new; no 0.10.
arXiv last-14-days: *SemanticPlan* (2607.04331, nuPlan), *CLEAR* (2607.02841, CARLA,
datacenter-scale), *Bench2Drive-Robust*, *HiDrive* — none a small-team-usable tooling
drop. Dataset streaming unchanged: **litdata** stays the best low-overhead fit for a
small team streaming shards to Colab/A40; nothing new to adopt.

## 3. Resource declaration (gate G-I)

- **Resource used:** local RTX-4060 dev box only (git operations + pytest, ~6 s of CPU
  total across the falsifier suite + 3 live-tree runs) + web searches for the sweep.
- **Wall-clock:** ~1.6 h. **Cost:** $0.
- **Why not the eval pod / Colab:** this run's experiment is a *git/filesystem* tool —
  it has no tensor workload, so an A40 would sit idle. The eval-pod-worthy item this run
  *surfaces* is the AlpaSim single-A40 harness smoke test (new backlog P1) — packaged as
  the next-run job, not run now (no owned pod slot this session; pod2 is no-touch,
  pod3/REF-A idle-usable but the AlpaSim image pull + NuRec render is a multi-hour job to
  scope first).

## 4. Actionable recommendations (G-B)

1. **Every agent runs `session_guard` at session end** (now protocol-mandated). Orchestrator:
   act on this run's WARN list — 9 stranded branches + 5 stale INTAKEs are ready to merge/triage.
2. **Lock the edge target chip before any quantization work** (Architecture/Prod-Opt): Orin →
   FP8/INT8 + per-layer ViT benchmark (INT8 can regress); Thor → NVFP4 via Edge-LLM. Ties H-C2/P5.
3. **AlpaSim single-A40 eval-harness smoke test** = next-run experiment (backlog P1). If it runs a
   NuRec rollout with a small policy in an A40's VRAM, it becomes our closed-loop eval candidate
   alongside CARLA; if it needs multi-service multi-GPU, CARLA stays the Phase-0 closed-loop path.

## 5. Falsifier ledger (this run)

| Claim | Falsifier | Verdict |
|---|---|---|
| The guard catches real stranded work | Run on live tree; must flag the review's debt classes | **PASS** (5 hub / 9 branch / 5 INTAKE) |
| No false positives on non-hub files | `tools/` untracked files must not block; unit test | **PASS** |
| Porcelain first-line parse is correct | Modified tracked hub file must BLOCK | **PASS** (caught+fixed pre-ship) |
| AlpaGym usable in our envelope | Docs specify GPU count for default policy | **REFUTED** (2-GPU min → reference only) |
| Orin can use NVFP4 for 4× win | TensorRT Edge-LLM runtime chip support | **REFUTED** (Thor-only; Orin = FP8/INT8) |
