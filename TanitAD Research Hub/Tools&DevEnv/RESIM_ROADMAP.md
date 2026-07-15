# TanitResim — Product Roadmap (P1, own continuously)

> TanitResim = the researcher-facing replay/eval product (`stack/tanitad/resim/` +
> `stack/scripts/resim_app.py`, single-port FastAPI, pod-servable). D-029 made it a first-class,
> continuously-developed product: **every Tools&DevEnv run measures a real gap, improves it, tests
> it, and pushes to `main`** (dev-tooling exception to the intake rule — commit under `resim:`).
> This file is the living roadmap; refresh it each run. Created 2026-07-15 (was mission-named but
> missing).

## Current state (as of 2026-07-15)
- Web app shipped (`d9ad898`): scenario cards, side-by-side arm panels, single-port serving.
- Camera-canvas aspect + fan-draw projection bug fixed (`b9a5ce9`); per-corpus overlay projection
  (physicalai f=444/h=1.43) fixed the urban fan pointing at sky (`b06222a`).
- Formal D1–D3 gates unified into the resim UI + `replay_app` (`827805d`, `97edaaf`), DRY-shared
  with `compare_arms`.
- Tests: `test_resim.py`, `test_replay.py` (green in the full suite).

## Known bugs (from mission brief + observed)
- **dual-sink (serve+rrd) writes an empty file** — the `.rrd` sink produces 0 bytes when serving
  live simultaneously. Repro + fix needed.
- **live-proxy gRPC path** — hosted-viewer path over the RunPod proxy is fragile.

## Roadmap (priority order; each item ships with a measured before/after)
1. **3-arm view** — flagship / REF-A / REF-B side-by-side, now that REF-B has landed (the reset
   `refb-speed-30k` arm). *Measure:* episode load time + a 3-panel screenshot. **← next-run P0.**
2. **Checkpoint A/B diff** — load two checkpoints of the same arm, diff predicted trajectories /
   gate numbers side by side. Directly serves the flagship-vs-reset comparison.
3. **Per-scenario filtering** — filter episodes by SC-01…SC-14 scenario label (co-own the label
   column with Opponent Analyzer / TanitScena).
4. **Worst-K reel** — auto-surface the K episodes with the largest predicted-vs-actual ADE — the
   "show me the failures" view researchers actually want.
5. **Latency / CNCE panel** — surface the I8 tick-latency + CNCE numbers per checkpoint (from the
   Prod-Opt latency baseline) so efficiency is visible in the same product.
6. **Export-to-figure** — one click → publication-ready figure for the paper (`Paper/`).

## Design language
Same as TanitScena (P2) — shared static/CSS idiom; single-port FastAPI, pod-servable, no external
CDN. Keep the encoder input static `[6,256,256]` so sim-eval and deployment never diverge.

## Standing rule
A run that ships a cross-agent instrument (like the 2026-07-15 CI gate) may skip TanitResim once,
but **not twice** — if TanitResim is untouched two runs running, the next run's P0 is a TanitResim
gap, no exceptions.
