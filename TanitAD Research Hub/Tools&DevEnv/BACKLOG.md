# Tools & DevEnv — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **TanitResim P1 measured gap (D-029 continuous product)** — pick ONE real researcher pain point
   from `RESIM_ROADMAP.md` and ship it push-to-`main`: **3-arm view** (REF-B has now landed —
   flagship/REF-A/REF-B side-by-side) OR **checkpoint A/B diff** panel. Measure: load time for a real
   episode + one before/after screenshot. This discipline's #1 standing duty went untouched
   2026-07-15; do not skip two runs.
2. **`episode → Rerun .rrd` replay/viz (backlog duty #2)** — predicted-vs-actual trajectory + BEV
   overlay; doubles as the D3 imagined-vs-oracle visual. Note: the orchestrator already shipped a
   trajectory-fan overlay (`a25a3fe`) — SCOPE THIS as the *episode-replay* complement, don't dup.
   `pip install rerun-sdk`; measure setup cost + one real episode → .rrd size/time for G-T1.
3. **`ci_check.py --self-test` mode** — monkeypatch a batch-stat layer into a throwaway module and
   assert CI exits 2, closing the I2-red-live falsifier (currently unit-tested only). Small, high
   confidence. Also: wire `-WarmBudgetS 35` into the loop's commit path once the full suite > 30 s.

## P1

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
- (2026-07-15) **CI commit gate `ci.ps1` (was P0 #1) DONE** — I2 tripwire (fail-fast) + per-test
  latency budget (>6 s → block, falsifier CONFIRMED) + suite-green + opt-in warm-wall. Unit-tested
  core `ci_check.py` (12 tests). Measured warm wall: quick gate **8.3 s** / full gate **24.4 s** (363
  tests). 0 new deps. Committed to `stack/` (D-029 tooling exception, not intake). Old "<15 s full
  suite" target retired as stale (suite grew 181→351). `2026-07-15-ci-commit-gate-and-latency-budget.md`.
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
