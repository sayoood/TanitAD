# PLAN — W5 visual reporting (priority order, so a killed run still yields value)

Pre-registered in `SPEC.md` (blob `fa444902…`, amendment A1 `0431fabd…` — the orchestrator's rename to
`benchreport` + W1 schema v1, both recorded BEFORE any rendering code). Order below is what was
executed; each step banks before the next starts.

| # | step | state |
|---|---|---|
| P0 | read the contract (`BUILD_PLAN.md` §0/§1/§2-W5/§3), the binding rules (four families, tier/loop, traps), and the `dataviz` skill; run its palette validator in BOTH modes and bank the output | ✅ `raw/palette_validation.txt` |
| P1 | **fixture adapters** E2/E1 raw → a §1 run directory, every number recomputed from the devkit CSVs and cross-checked against E2's own `scores_summary.json` | ✅ `benchreport/adapt.py`; xcheck PASS (≤ 1e-12) |
| P2 | **contract reader** (`contract.py`) + W1 schema validation + the cross-field rules the schema cannot express (floors are arms, every arm paired against every floor, tier/loop present) | ✅ |
| P3 | **the page**: header stamp on every screen AND printed page, banners, KPI tiles, headline vs floors + published (same protocol only), paired W/T/L, sub-metrics per stage + multiplicative zeros, per-log, per-speed-band, four families | ✅ `render.py`, `charts.py`, `svg.py`, `page.py`, `palette.py` |
| P4 | **parse-back verification** written independently of the renderer + the two mutations that must go RED | ✅ `verify.py`, `tests/test_benchreport_golden.py` |
| P5 | **failure gallery**: worst-scene selection here, nuPlan map + agent boxes + plans + camera projection in the NavSim venv (py3.9), text overlay with command / decoded manoeuvre / failing sub-metrics | ✅ `gallery.py`, `navsim_gallery.py`, `camproj.py` |
| P6 | **leaderboard chart components for W4** (`render_leaderboard_charts(model)`) | ✅ `leaderboard_charts.py` |
| P7 | showcase rendered from E2's real outputs and banked; docs; stage | ✅ `raw/showcase_e2_warmup/` |

**Deliberately NOT done** (named, not silently dropped): rendering a stage-1 scene in the gallery (the
NavSim-side renderer reads synthetic stage-2 pickles only — stage-1 scenes come from the log pickles,
a W1/W3 input once navhard lands); a nuScenes-specific panel set (W6's protocol has no run yet — the
report falls back to the generic path and says so); `--single-file` HTML embedding (the run directory
is the unit that travels).
