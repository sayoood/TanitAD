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

## P0 — next run

1. **Make `ci_gate` + `session_guard` + `fleet_probe` unskippable (session/cron hook wiring)
   — now the top item, and it is the same gap three times.** All three are disciplines an
   agent must remember to perform; nothing executes any of them automatically. `fleet_probe`
   raises the stakes: a probe nobody runs is exactly as blind as the grep-based monitor it
   replaced, and GOALS G1's "detected within one 6-hour cycle" is unprovable until a cron runs
   it and pages on exit code 2. Method: pre-push/session-end hook for the two gates + a
   6-hourly cron for the probe. Falsifiers: (a) a deliberately-red branch must be un-pushable
   without an explicit override flag; (b) a trainer killed by hand must produce a RED alert
   without any human invoking the probe.
2. **`rr_log.py` dual-sink tee (findings-driven; replaces the retired viz item).** Two
   `RecordingStream`s + an explicit `recording=` on every `rr.log`, because
   `rr.set_sinks(FileSink, GrpcSink(url))` after `serve_grpc()` **deadlocks** (measured
   2026-07-21: killed at 120 s, no output — the sink connects back to its own in-process
   server). Resource: CPU, ~2 h. **Pre-registered falsifier: the dual-sink `.rrd` must land
   within 5 % of the single-sink `.rrd` (52,966 B/window baseline) for identical input** —
   anything smaller is still a stub. Until then the shipped guard refuses the combination.
3. **Pin `rerun-sdk==0.34.1`** — the entire viz backbone depends on it and it is pinned in
   **no** requirements file in the repo (measured 2026-07-21). Trivial; blast radius is not.
4. **Wire the Rerun 0.34.1 Viewer-MCP into an agent tool list** — the surviving half of the
   old P0#1. The rest of that item was stale and is retired: 0.34.1 was **already installed**
   and `rr_log.py` (417 lines) **already logs episodes**, so the "pin + migrate, 1–2 h"
   work did not exist. This is the GOALS G2 lever that turns "the overlay looks right" from an
   assertion into a verification. Expected ~30 min; measure it (G-T1).
5. **`rrd_bench` on a real episode** — current numbers are synthetic records; confirm
   B/window at true frame entropy on one `ep_*.pt`. Falsifier: if real-frame B/window differs
   >2× from 52,966, the synthetic baseline is not a valid stand-in and G2's 5 % test re-bases.
6. **`gpu_tripwire` v2 — bf16/AMP arm + CUDA-graph capture probe.** v1 is fp32 + eager only,
   so Prod-Opt's CUDA-graph deploy tick (`b984e04`, 11.16 ms) and every bf16 training path are
   still unguarded. Method: add a bf16 autocast parity arm (looser tol, measured first) and a
   capture/replay-equivalence probe. Resource: 4060, minutes. Expected: bf16 deviation ~1e-2
   on O(1) activations — **measure before setting the tolerance**, do not guess it.
7. **Re-scoped: `test_replay_app_test_mode_and_regression_gate`** — the "10.86 s tall pole" was
   partly an I/O+contention artifact: measured **8.02 s clean / 14.90 s beside a second pytest
   process** (2026-07-20). Two questions now, not one: (a) can the FastAPI TestClient boot be
   shared across the module (fixture reuse)? (b) should `ci_gate` detect concurrent load and
   widen the per-test budget rather than false-positive? Falsifier for (a): if it stays >6 s
   after fixture reuse, the cost is the boot, not the payload → keep 15 s and document.

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
