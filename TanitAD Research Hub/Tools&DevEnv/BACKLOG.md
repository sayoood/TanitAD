# Tools & DevEnv — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **`episode → Rerun .rrd` replay/viz (backlog duty #2)** — predicted-vs-actual trajectory + BEV
   overlay; doubles as the D3 imagined-vs-oracle visual. Note: the orchestrator already shipped a
   trajectory-fan overlay (`a25a3fe`) — SCOPE THIS as the *episode-replay* complement, don't dup.
   `pip install rerun-sdk`; measure setup cost + one real episode → .rrd size/time for G-T1.
2. **INT8-on-target export verification (findings-driven, 2026-07-11 note §2)** — the TensorRT Q/DQ
   trap makes an RTX-clean INT8 export NOT imply an Orin/Thor-clean engine. Build a tiny checker that
   scans an exported ONNX for **reshape-fed Q/DQ scales** (the failure pattern) and a `trtexec`
   build-smoke that MUST run on-target (Orin/Thor or the graphics-pod), not the 4060. Pairs with
   Prod-Opt's `int8_quant/`. Expected: the checker flags Dynamo-int8 exports; falsifier: a
   direct-fp32-initializer export passes both x86 and (when available) ARM build. BLOCKED on an ARM
   target for the trtexec half; the ONNX-scan half runs now on the 4060.

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
- (2026-07-11) **CI script `ci.ps1` (was P0 #1 / backlog duty #3) DONE** — I2 tripwire + pytest via
  `profile_testsuite.py check`; measured PASS 17.2 s (I2 2.4 s + suite 14.8 s, 189 passed, overhead
  1.43 s); falsifier holds (7.0 s test → exit 1). Intake `2026-07-11-ci-gate/` (ci.ps1 +
  ci_i2_tripwire.py + 3 tests). Pending triage; chains on the profiler intake below.
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
