# SPEC — W5: publication-quality per-run reports for the TanitEval bench suite

> **AMENDMENT A1 (2026-09-19 ~13:47Z, orchestrator ruling, BEFORE any rendering code or any
> rendered number).** (1) The package is **`taniteval/taniteval/benchreport/`**, CLI
> **`python -m taniteval.benchreport <run_dir>`**, tests **`taniteval/tests/test_benchreport_*.py`**
> — a `report/` package would shadow the legacy `taniteval/taniteval/report.py`, which W5 never
> touches. Every `taniteval.report` / `report/**` / `test_report_*` below reads as the renamed path;
> truth rule 6 becomes: *the legacy importers pass before AND after W5* (27 / 27, banked in
> `tests/legacy_importers_before_after.txt`). (2) W1's schema v1 is frozen on disk
> (`taniteval/taniteval/bench/schema/{bench_run,summary}.schema.json`); the E1/E2 adapters emit
> **schema-valid** files, asserted in a test with W1's own `schema_check.validate`. W4 imports the
> chart components from `taniteval.benchreport` (`leaderboard_charts.render_leaderboard_charts`).
> The original pre-registration hash stays in `raw/SPEC_PREREG_HASH.txt`; this amendment's hash is
> appended there.

**Work package** W5 of `2026-09-19-eval-suite-build/BUILD_PLAN.md` · **owner** TanitAD_EvalFlyWheel ·
**commissioned** by the PI in chat 2026-09-19 (*"I want you also to add high quality visualizations and
reporting to the eval suite"*; *"start now and take the risk"*) · **owns (exclusive write)**
`taniteval/taniteval/report/**`, `taniteval/tests/test_report_*.py`, this package.
**Pre-registered before any report code was written** (hash in `raw/SPEC_PREREG_HASH.txt`).

## 0. The deliverable

```
python -m taniteval.report <run_dir>            # writes <run_dir>/report/index.html + report/fig/*.svg|png
```

A run directory is the BUILD_PLAN §1 contract (`bench_run.json`, `summary.json`, `scores/<arm>.csv`,
`artifacts/<arm>.json`, `criteria/<arm>.txt`). W1 owns that schema and it does not exist yet, so W5
ships an **adapter** that turns E2's (and E1's) existing raw outputs into a §1 run directory, and reads
§1 through ONE module (`report/contract.py`) so switching to W1's schema is a one-file change.

## 1. What every report shows (each is an acceptance item)

| # | content | rule it enforces |
|---|---|---|
| H | a header on **every page** (screen and every printed page): protocol tag, devkit SHA, tier + loop stamp, evidence class | `CLAUDE.md` "EVERY NUMBER CARRIES ITS EVAL TIER"; §1 "tier + loop stamp on every number" |
| a | headline per arm against the **mandatory floors** (STOP, CV; ECHO when present) drawn as reference lines, plus the **published results for the SAME protocol only** | "never two protocols on one axis" (`navsim.cross_protocol`); a chart REFUSES rows of a second protocol (raises) |
| b | sub-metric breakdown **per stage** (EPDMS: NC DAC DDC TLC EP TTC LK HC EC; PDMS v1: NC DAC EP TTC C) with the multiplicative-zero rates | a stage an arm did not produce (e.g. a CV stand-in) is shown as a refusal with reason + n, never as the arm's number |
| c | per-log and per-speed-band panels; paired win / tie / loss vs **CV** and **STOP** | both floors MANDATORY: a run lacking one is bannered **INCOMPLETE** with the missing floor named |
| d | four-family panels (LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC) | a refused family / metric is shown with its reason and n, never silently dropped |
| e | failure gallery of the worst scenes: BEV on the real nuPlan map (lanes, drivable area, agent boxes) with plan vs human vs CV vs STOP, beside the stitched 3-camera frame with the plan projected, text overlay (command, decoded manoeuvre, failing sub-metrics) | map/BEV drawn by the NavSim devkit's own `navsim.visualization` in the NavSim venv (py3.9); the decoded manoeuvre is the programme's canonical factored labeller (`four_families.maneuver_kinematics` + `refc_tactical.factor_from_kinematics`), not a new gate |
| f | leaderboard chart components for W4: rank bars per protocol, params-vs-score scatter, our rows highlighted, floors as reference lines | same one-protocol-per-axis refusal |

Form rules (loaded `dataviz` skill): palette = the skill's validated reference instance (validator run
per mode, output banked); marks thin; legends for ≥ 2 series; every chart has a table-view twin; a
per-mark tooltip; light AND dark selected (not flipped); print-clean. Self-contained: inline CSS and JS,
no CDN, no web fonts; raster gallery figures are files under `report/fig/`.

## 2. The truth rules (each is a test)

1. **The report prints the SAME numbers as `summary.json`, verified by PARSING the HTML.** Every number
   is emitted with `data-k` (its key path in summary.json) and `data-v` (full precision); a verifier
   written independently of the renderer resolves every `data-k` against summary.json with its own
   reader, requires `data-v == value` exactly and the visible text == the value at the rendered precision,
   and requires the arm LABEL shown beside each number to be that arm's label. The CLI runs it on its own
   output and exits non-zero on any mismatch (`report/verify.json` is the artifact).
2. **Literal goldens from an independently authored reference** — E2's `RESULT.md` (written by E2, not by
   this renderer): S2-EPDMS-u A1 **0.4670**, A2 **0.5211**, A3 **0.4546**, A4 **0.0950**, CV **0.3971**,
   ECHO **0.4287**, STOP **0.5212**; official two-stage EPDMS CV **0.1854**, ECHO **0.2248**, A4
   **0.0000**, STOP **0.3009**; W/T/L A1 vs CV **51 / 76 / 77**, A1 vs STOP **108 / 40 / 56**; A1 − CV by
   t0 speed band (n): <1 m/s (36) **+0.031**, 1–4 (64) **+0.100**, 4–8 (59) **+0.151**, ≥8 (45)
   **−0.052**. Written as LITERALS in the test, never as expressions over the code under test.
3. **Mutation that must go RED:** a swapped arm label in the renderer (A1 ↔ STOP) must fail the verifier;
   a second mutation (a value read from the wrong arm) must fail it too.
4. **Adapter cross-check:** every field the adapter recomputes from the devkit CSVs that E2 also computed
   (`raw/scores_summary.json`, E2's own `parse_scores.py`) must agree to ≤ 1e-12; else the adapter refuses.
5. **Projection identities (analytic):** a point on the virtual boresight projects to the frame centre
   ((W−1)/2, (H−1)/2) exactly; a ray from `tanitad.data.calib.cylindrical_rays` at pixel (u, v), pushed
   into the world and projected back, returns (u, v) to < 1e-6 px.
6. **Legacy compatibility:** `taniteval/taniteval/report.py` is shadowed by the new package; its public
   and private API (`build`, `_lb_rows`, `_primary`, …) must stay reachable as `taniteval.report.<name>`
   (`tests/test_estimator_closeout.py`: 27 passed BEFORE the package existed — must stay 27 passed).

## 3. Showcase

One real report rendered from E2's warmup outputs, kept in `raw/showcase_e2_warmup/` (run dir +
`report/`), figures < 20 MB total. It is a FIXTURE conversion, not a bench run: `bench_run.json` carries
`"leaderboard_eligible": false` so W4 never double-counts it against W1's canonical run.

## 4. Out of scope / refused here

Submission of anything (PI: none until approved) · GPU (rendering is CPU) · editing W1–W4/W6 files ·
re-reading published numbers from PDFs (W4's job; W5 uses a FIXTURE built from `NAVSIM_PROTOCOL.md`,
evidence class INHERITED, every value checked against the line it came from).
