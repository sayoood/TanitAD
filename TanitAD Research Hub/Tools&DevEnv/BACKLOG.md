# Tools & DevEnv — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

1. **`episode → Rerun .rrd` replay/viz (backlog duty #2)** — predicted-vs-actual trajectory + BEV
   overlay; doubles as the D3 imagined-vs-oracle visual. Note: the orchestrator already shipped a
   trajectory-fan overlay (`a25a3fe`) — SCOPE THIS as the *episode-replay* complement, don't dup.
   **Rerun 0.34.1 confirmed current (2026-07-07 lit sweep), G-T1 GO ~15 min**; the 0.34 Viewer-MCP
   also lets an agent drive the replay. `pip install -U rerun-sdk`; measure setup cost + one real
   episode → .rrd size/time for G-T1. **This is the top NEXT increment.**
2. **INT8-on-target export verification (findings-driven, 2026-07-11 note §2)** — the TensorRT Q/DQ
   trap makes an RTX-clean INT8 export NOT imply an Orin/Thor-clean engine. Build a tiny checker that
   scans an exported ONNX for **reshape-fed Q/DQ scales** (the failure pattern) and a `trtexec`
   build-smoke that MUST run on-target (Orin/Thor or the graphics-pod), not the 4060. Pairs with
   Prod-Opt's `int8_quant/`. Expected: the checker flags Dynamo-int8 exports; falsifier: a
   direct-fp32-initializer export passes both x86 and (when available) ARM build. BLOCKED on an ARM
   target for the trtexec half; the ONNX-scan half runs now on the 4060.

## P1

2b. **Trackio experiment tracker spike (findings-driven, 2026-07-11 lit sweep)** — HF Trackio is a
   local-first **W&B drop-in** (`import trackio as wandb`), $0, no lock-in, and shims AlpaGym's W&B
   dep for the Phase-1 spike. Do a `hello-run`: log a real training run's loss/erank/step_ratio,
   measure setup minutes + dashboard usability, verdict vs our current ad-hoc JSON logs. G-T1 GO
   ~10 min expected. Falsifier: if the drop-in shim breaks on our log calls or needs a server, NO-GO.
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
7. **AlpaSim/AlpaGym clone-and-inspect** (findings-driven) — both `NVlabs/alpasim` + `NVlabs/alpagym`
   now public. VRAM floor updated to **2×24 GB** with an official local smoke config
   `experiment=alpamayo_1_5_local_2gpu_smoke` (was "40–60 GB single"). Read for a lighter reference
   policy / harness we could adapt Phase-1 (NOT a Phase-0 adopt). Deliverable: a Phase-1 adoption
   note w/ the concrete integration surface + VRAM measured on a scoped 2×GPU RunPod spike; treat
   10 B Alpamayo as an oracle/data source. Watch Cosmos-RL as a borrowable scorer.

## Handoffs to other disciplines
- **→ Benchmarks & Eval (Thu):** **Bench2Drive-Robust** (2605.18059) adds a compute-induced
  **inference-delay** perturbation axis to closed-loop E2E-AD eval — scores exactly the low-latency
  regime that is our edge advantage. Recommend adding the delay axis to the eval plan so latency
  headroom becomes a rewarded metric (C2). Companions Bench2Drive-VL / -Speed.
- **→ Production & Optimization (Sat):** on-target INT8 build (TensorRT Q/DQ trap, note §2) — build
  the `int8_quant/` engine with `trtexec` on a real Orin/Thor, not only the 4060; pre-scan the ONNX
  for reshape-fed Q/DQ scales. **ZipDepth** (6.1 M, 2607.08771) is a candidate cheap on-Orin depth
  aux signal (H16).

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
