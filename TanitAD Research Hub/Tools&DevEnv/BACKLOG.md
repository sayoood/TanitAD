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
2. **ci_gate extension:** fold the newly-merged suites into the one-command gate — test_lake (9),
   test_refa_flagship_parity, test_eval_behavior (22), the calib trio (test_calib + test_calib_r1
   + test_physicalai_rig as ONE unit), the metric-suite tests (22). Report the new green total +
   wall-clock; keep it under 5 min or shard it.
3. **Colab job-card bootstrap (mandate M-1.3 enabler):** `tools/colab_job_template.py` — data-pull
   cell (HF gated, token from Keys.txt read-in-place), run cell, results-push cell (back to the
   repo Implementation/ dir) + a README teaching the pattern. Prove it end-to-end with one real
   job (e.g. a probe fit on cached latents). Every agent's M-3 escalation path depends on this.
4. **Eval-pod access hygiene:** a `tools/evalpod.md` one-pager (SSH alias, TanitEval CLI, the
   LOCK touch-file convention, the pod2 no-touch rule, the memory-safe ckpt relay pattern) so
   every agent can use the pod without re-learning the ops constraints from incident history.

## P0 — next run

1. **`episode → Rerun .rrd` replay/viz (backlog duty #2)** — predicted-vs-actual trajectory + BEV
   overlay; doubles as the D3 imagined-vs-oracle visual. Note: the orchestrator already shipped a
   trajectory-fan overlay (`a25a3fe`) — SCOPE THIS as the *episode-replay* complement, don't dup.
   `pip install rerun-sdk`; measure setup cost + one real episode → .rrd size/time for G-T1.
2. **Speed up `test_replay_app_test_mode_and_regression_gate` (10.86 s, the suite tall pole)** —
   ≈20–23 % of total wall; the one test worth optimizing. Goal: get it under ~4 s (fixture reuse /
   smaller synthetic bundle) so `ci_gate`'s per-test budget can tighten from 15 s toward the original
   6 s intent. Falsifier: if it stays >6 s after the split, the cost is the FastAPI TestClient boot,
   not the payload → keep 15 s and document.

## P1

0. **AlpaSim single-A40 eval-harness smoke test** (findings-driven, 2026-07-18 note §2). `NVlabs/alpasim`
   is now public (Apache-2.0). Method: on an idle A40 pod (pod3/REF-A between runs), clone, pull one
   `PhysicalAI-AV-NuRec` scene, run the NuRec renderer + runtime with a small policy for a short
   rollout. Resource: 1×A40; scope the image-pull + render cost FIRST (multi-service, GPU-heavy).
   Expected: if a rollout fits in one A40's VRAM → AlpaSim becomes a closed-loop eval candidate
   alongside CARLA; if it needs multi-service multi-GPU → CARLA stays the Phase-0 path. Falsifier:
   default no-go if a single-scene rollout can't run under 48 GB. AlpaGym (RL) is 2-GPU-gated →
   explicitly NOT in scope. Deliver as a job card (Colab won't fit; A40 only).
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
