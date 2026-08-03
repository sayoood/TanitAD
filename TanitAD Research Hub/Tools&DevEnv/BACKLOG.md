# Tools & DevEnv — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — FLEET DIRECTIVE 2026-07-17 (Sayed; supersedes prior P0 ordering; resource-mandated G-I)

Context: `Project Steering/FLEET_REVIEW_2026-07-17.md`. The review merged 5 stranded branches
(~15k lines) — that debt class is now YOURS to prevent structurally.

1. ~~**D-026 guardrail**~~ **DONE (2026-07-18)** — `tools/session_guard.py` + `.ps1` + README + 15
   falsifiers; protocol-wired into G-F (BLOCK on uncommitted hub deliverables) / G-I (WARN list =
   stranded-branch check). Live-tree run flagged 5 hub files / 9 branches / 5 stale INTAKEs. Follow-ups
   (new P0.1a): (i) wire into a real session-end hook so it can't be skipped; (ii) an `--open-merge`
   mode the ORCHESTRATOR (not agents) runs to auto-land the WARNed branches; (iii) fold the WARN list
   into the orchestrator's weekly triage input.
2. ~~**ci_gate extension**~~ **DONE (2026-07-20)** — `tools/ci_gate.py` v2: SUITE_MANIFEST
   (16 modules pinned to a collected-count floor), `--min-total 390`, `--gpu-smoke`, `--json`;
   skips green unless a whole module is skipped. Both trees GATE PASS: **396/39.0 s** worktree,
   **531/60.2 s** Drive. **Sharding NOT needed — 5x under the 5-min ceiling.** Backlog text was
   stale: `test_eval_behavior` is 13 (not 22) and **`test_calib_r1.py` does not exist** (folded
   into `test_calib`). Follow-ups → new P0.2/P0.3 below.
3. **Colab job-card bootstrap (mandate M-1.3 enabler):** `tools/colab_job_template.py` — data-pull
   cell (HF gated, token from Keys.txt read-in-place), run cell, results-push cell (back to the
   repo Implementation/ dir) + a README teaching the pattern. Prove it end-to-end with one real
   job (e.g. a probe fit on cached latents). Every agent's M-3 escalation path depends on this.
4. **Eval-pod access hygiene:** a `tools/evalpod.md` one-pager (SSH alias, TanitEval CLI, the
   LOCK touch-file convention, the pod2 no-touch rule, the memory-safe ckpt relay pattern) so
   every agent can use the pod without re-learning the ops constraints from incident history.

## P0 — next run (re-prioritized 2026-08-03, first DAILY slot)

0. **`RESIM_ROADMAP.md` — WRITE IT.** Mission P1 says the TanitResim roadmap lives there and it
   still does not exist; this is the FOURTH run carrying it, each time losing to more urgent work.
   That is exactly why it goes first now. Scope for one day: the gap list already known
   (dual-sink empty file, live-proxy gRPC, 3-arm view, per-scenario filtering, worst-K reel,
   checkpoint A/B diff, latency/CNCE panel, export-to-figure), each with a measured cost estimate
   and a go/no-go under P5 (G-T1). Resource: 0-GPU. Falsifier: if writing it surfaces no concrete
   next increment, the product mandate itself needs renegotiating with Sayed — say so.
5. **Thor GPU is UNMONITORED (findings-driven, 2026-08-03).** `nvidia-smi --query-gpu=...` returns
   nothing on Jetson — it exposes `tegrastats`. `fleet_probe` therefore reports Thor `AMBER
   NO_GPU_READOUT`, which is honest but blind, and becomes a real gap the moment Thor runs
   inference. Method: add a `tegrastats`-based GPU readout for `role=edge` hosts (one sample,
   parse `GR3D_FREQ`), behind the same absence-is-an-alarm rule. Resource: Thor over ssh, minutes.
   Expected: util + RAM on the same table row as the pods. Falsifier: if `tegrastats` needs root
   or blocks, record the blocker and keep the honest AMBER rather than inventing a green.
6. **`JOB_RE` is the THIRD level of the stale-name defect (findings-driven, 2026-08-03).** The
   class has now appeared at log names (v1) and host names (v2); `JOB_RE` still encodes what a
   trainer *cmdline* looks like, so a run launched through an unmatched wrapper is invisible the
   same way. Method: cross-check discovered jobs against GPU-owning pids (`nvidia-smi
   --query-compute-apps`) — a pid holding GPU memory that `JOB_RE` did not classify is an alarm,
   and that check cannot go stale because it does not name anything. Resource: 0-GPU + one live
   probe. Expected: 0 unclassified GPU owners on a healthy pod. Falsifier: if `--query-compute-apps`
   is empty inside these containers (it often is under some runtimes), the check is not available —
   measure that first and say so rather than shipping a check that always passes.
1. **`episode → Rerun .rrd` replay/viz + the 0.34.1 Viewer-MCP upgrade (duty #2)** — unchanged
   from 2026-07-20/21; predicted-vs-actual trajectory + BEV overlay, doubles as the D3
   imagined-vs-oracle visual. `rerun-sdk==0.34.1` is already pinned in the venv.
2. **Make `ci_gate` (and now `fleet_probe`) unskippable — session/pre-push hook wiring.** Nothing
   executes either automatically; both are disciplines an agent must remember. One wiring closes
   both. Falsifier: after wiring, a deliberately-red branch must be un-pushable without an explicit
   override flag.
3. **`gpu_tripwire` v2 — bf16/AMP arm + CUDA-graph capture probe.** v1 is fp32 + eager only.
   **Measure the bf16 deviation before setting its tolerance** — do not guess it.
4. **Re-scoped: `test_replay_app_test_mode_and_regression_gate`** — 8.02 s clean / 14.90 s under
   contention. (a) can the FastAPI TestClient boot be shared across the module? (b) should
   `ci_gate` detect concurrent load rather than false-positive?

## P1

0. ~~**AlpaSim single-A40 eval-harness smoke test**~~ **RETIRED (2026-07-20) — answered NO-GO by
   another agent on 2026-07-19**, before I got to it. The eval pod is itself an unprivileged
   container with **no nested container runtime**, and AlpaSim's NuRec renderer ships only as
   `nvcr.io/nvidia/nre/nre-ga:26.04` (no source form). Policy side GO (bare gRPC, adapter
   written); ~1.5 GB/scene + <2 GB VRAM would fit a proper host. Residual ask = **infra: a
   docker-capable GPU host** (Sayed decision, → "Blocked on Sayed"), not a tooling task. See
   `Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/INTAKE.md`.
0b. **Watch TerraZero for a code release** (arXiv 2607.13028, Applied Intuition) — procedural
   driving sim at 1.3 M agent-steps/s on ONE GPU, no rendering: the closed-loop harness shape
   our envelope can actually afford, and the natural fallback now AlpaSim is infra-blocked.
   No code today (commercial vendor → assume closed). Cost to check: ~5 min/run. If code lands,
   promote to P0 immediately: integration est. 1–2 days.
3. **CARLA graphics-pod recipe — dry-run when a graphics GPU is available** (findings-driven, note
   §1). On any graphics-capable pod: verify `vulkaninfo | grep deviceName` returns the GPU, then
   `Xvfb :99 + CarlaUE4.sh -RenderOffScreen`; measure boot-to-first-rendered-frame + a 100-tick
   camera rollout. Gate for checkpoint-driven ego eval in CARLA. BLOCKED on a graphics pod (Sayed);
   NOT urgent (milestone 1 needs no pixels). Expected: first RGB frame < 60 s after server up.
4. **Pod bootstrap script v2** — one-command environment restore for a NEW pod (apt, venv, repo,
   epcache warm, Colab-CLI); measured restore time. Resilience for "pod died, new ssh".
5. **Verify the Drive "Available offline" fix** (needs Sayed to pin `stack/` first) — re-measure cold
   suite; expected cold ≈ warm (~10.7 s) i.e. ~30 s saved/run. Falsifier: if cold stays >30 s after
   pinning, hydration is not the cause → re-open. (Tool ready: `profile_testsuite.py profile`.)

## P2

6. **Windows/Linux path+encoding audit tooling** — the `|`-in-filenames and mojibake classes;
   a lint script for non-NTFS-safe names and non-UTF8 writes in the repo.
7. **AlpaSim clone-and-inspect** (findings-driven) — `NVlabs/alpasim` is now public. Read the repo
   for a lighter reference policy / harness we could adapt Phase-1 (NOT a Phase-0 adopt; 40–60 GB
   VRAM). Deliverable: a Phase-1 adoption note with the concrete integration surface + VRAM measured.

## Done / retired
- (2026-08-03) **`fleet_probe` v2 — DISCOVERY-BASED MEMBERSHIP + `/proc/fd/1` log binding DONE**
  (`9584405`), and the 13-day-stranded v1 landed (`7f34086`). MEASURED: the live ssh config holds
  **8 `tanitad-*` endpoints; v1's hardcoded dict knew 4**, and the missing ones included **pod5,
  running the flagship `v5f` at that moment** — v1 would have printed a complete-looking table
  with the only working host absent. `/proc/<pid>/fd/1` resolved the live flagship from
  `AMBER NO_LOG_BOUND` to **GREEN step=1250 log_age=51 s** without loosening any check. Two of my
  own v2 defects fixed from the live run (a manufactured `RED DISK_FULL` at a guessed path;
  thor-wifi as a phantom second host). 164 falsifiers / 38.3 s, discovery ones mutation-proven.
- (2026-07-21) **`tools/fleet_probe.py` DONE (unplanned — took the top slot because the
  program's #1 risk moved to ops).** Discovery-based fleet liveness: no hardcoded run/log
  names, absence of evidence is AMBER not GREEN. Live: 4 hosts in **9.7–11.3 s**; found pod2
  idle (RED) and pod3 unverifiable (AMBER); 20 falsifiers 0.35 s. `.claude/skills/
  fleet-status/SKILL.md` rewritten to call it. Follow-up → new P0#1 (cron it).
- (2026-07-21) **rerun `.rrd` measured (old P0#1's real content)** — 52,966 B/window at
  jpeg85, 299 win/s; **dual-sink = 3,196 B stub, 3,314× loss**; guard shipped via intake
  `2026-07-21-rrd-dual-sink-guard/`. Item's migration premise was stale (see new P0#4).
- (2026-07-20) **ci_gate v2 + gpu_tripwire + session_guard source check DONE** — see P0#2 above.
  57 falsifiers 15.5 s; both trees GATE PASS; CUDA parity 4/4 on the 4060 (worst dev 9.5e-07,
  batch-1 encode 0.85–1.43 ms). The stranded `2026-07-17-ci-gate/` intake is **superseded** —
  `ci_gate` now lives at `tools/ci_gate.py` as repo-root dev tooling (same class as
  `session_guard`, no intake round-trip), and the intake carries a self-written verdict.
- (2026-07-17) **CI gate `ci.ps1`/`ci_gate.py` (was P0.1, backlog #3) DONE** — one-command self-testing
  gate; fails on failure/collection-error/slow-test/wall/missing-tripwire. 11/11 falsifiers; caught the
  live RED suite (exit 1, 3.9 s); clean 343+2skip 47–57 s. Intake `2026-07-17-ci-gate/`. Note: the
  original "wire to profile_testsuite.py + <15 s warm" target was stale (suite grew 181→343 tests, warm
  ~47 s); shipped as a standalone JUnit-based gate with a 15 s per-test / 90 s wall budget instead.
- (2026-07-09) **Test-suite I/O profiling (was P1.5) DONE** — cold 40.6 s / warm 10.7 s measured;
  root cause = Drive hydration latency; `profile_testsuite.py` shipped via intake (9 tests). Fix =
  pin `stack/` offline (→ new P1.5 verification item).
- (2026-07-09) **CARLA-on-pod harness (was P0.2)** — shipped LIVE by the orchestrator/loop
  (`carla_work_zone.py`, SC-01 measured in `-nullrhi`). This run added the camera-render root-cause
  + turnkey graphics-pod recipe (→ P1.3). Only the pixels path remains, gated on a graphics pod.
- (2026-07-09) **Colab CLI burst harness (was P0.1) DONE** — T4 validated end-to-end 33 s / $0
  (`Implementation/colab_burst/README.md`, commit `a604b21`).
- (2026-07-13) MetaDrive front-cam RGB + perturbation package shipped via intake; superseded by D-014.
- (2026-07-08) tmux removed from pod flow; detached setsid launcher + runner guard shipped (MVP).

## Blocked on Sayed
- MetaDrive supervised install — RETIRED by D-014 (CARLA replaces it). Removed from active backlog.
- Pin `stack/` to Drive "Available offline" (~1 click) → unblocks P1.5 verification, ~30 s/run G-E win.
- Graphics-capable pod recreation → unblocks P1.3 (CARLA camera pixels). NOT urgent.
- **A docker-capable GPU host** (nested container runtime) → the ONLY blocker between us and an
  AlpaSim closed loop; everything else is measured GO and the policy adapter is written
  (2026-07-19 investigation). Same infra class as the graphics-pod ask — worth deciding once.
