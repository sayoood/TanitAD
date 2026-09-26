# RESULT — W5: publication-quality reports for the one-command eval suite

**Stream** EvalFlyWheel W5 · **2026-09-19/20** · **Deliverable**
`python -m taniteval.benchreport <run_dir>` → `report/index.html` + `report/fig/*.svg|png` +
`report/verify.json`. Pre-registered `SPEC.md` (blob `fa444902…`; amendment A1 `0431fabd…`, the
orchestrator's rename + W1 schema v1, recorded **before any rendering code**;
`raw/SPEC_PREREG_HASH.txt`).

Every number in this document is **MEASURED** by this stream unless marked, with the artifact beside
it. The reports themselves carry the tier/loop/evidence stamp of the run they render (for the showcase:
**T1-family**, stage-1 loop **OPEN**, stage-2 **UNRULED**, IDM-reactive traffic, evidence **MEASURED**)
— W5 renders stamps, it does not mint them.

## 1. Findings first

1. ⭐ **The report proves itself.** Every rendered number carries its `summary.json` pointer; after
   writing the page the CLI PARSES ITS OWN HTML back with a verifier that shares no code with the
   renderer (own tree, own RFC-6901 resolver, own `%`-format table, own label lookup) and checks
   value, printed precision, the arm the number belongs to, the label shown beside it, every refusal,
   every published row's protocol tag, and coverage. **Showcase: PASS — 940 numbers and 46 refusals
   checked, 0 errors, 0 coverage gaps** (`raw/showcase_e2_warmup/report/verify.json`).
2. ⭐ **It reproduces E2's own numbers, from E2's own files.** The adapter recomputes everything from
   the devkit per-token CSVs and CROSS-CHECKS against E2's independently written `scores_summary.json`
   (≤ 1e-12, or it refuses). The rendered page then prints E2's RESULT.md literals exactly:
   S2-EPDMS-u **A1 0.4670 · A2 0.5211 · A3 0.4546 · A4 0.0950 · CV 0.3971 · ECHO 0.4287 · STOP 0.5212**;
   official two-stage EPDMS **CV 0.1854 (=18.5356 ×100, the HF warmup LB) · ECHO 0.2248 · STOP 0.3009 ·
   A4 0.0000**; paired **A1 vs CV 51/76/77**, **A1 vs STOP 108/40/56**; A1 − CV by t0 speed band
   **+0.031 (n 36) · +0.100 (64) · +0.151 (59) · −0.052 (45)**. Asserted as literals against E2's
   document AND against the HTML (`tests/test_benchreport_golden.py`).
3. ⭐ **The mutations go RED.** A swapped arm label (A1 ↔ STOP) → **230 verifier errors**; a value read
   from the wrong arm → RED on both `data-v` and the printed text; a per-token CSV cell moved by 1e-9 →
   the adapter **refuses** (`cross-check FAILED`). The unmutated control is PASS. ⚠️ Scope, measured
   while writing it: the cross-check does **not** cover the stage-1 rows of a HYBRID arm — those rows
   are the devkit CV stand-in's, and E2's summary publishes no stage-1 block for them.
4. ⭐ **The failure gallery draws the real map.** 8/8 worst scenes rendered: the nuPlan map (lanes,
   drivable area, walkways, intersections, baseline paths) and agent boxes drawn by the **devkit's own**
   `navsim.visualization` in the NavSim venv (py3.9), our plan against CV / STOP / ECHO, beside the
   **stitched 3-camera 256×640 cylindrical frame the model actually saw** with the same plans projected,
   and a text overlay carrying the NavSim command, the **decoded manoeuvre from the programme's own
   canonical labeller** (`refc_tactical.factor_from_kinematics`, never a second gate), the per-scene
   scores of every arm and the failing sub-metrics. `raw/showcase_e2_warmup/report/fig/gallery/`.
5. ⛔ **A HYBRID number is never printed.** The five camera arms have no admissible official EPDMS, so
   they are absent from that axis and listed under it with one shared reason; the devkit's hybrid rows
   render as `HYBRID` in the summary-rows table, not as numbers. (First draft printed them inside the
   refusal text — caught by reading the rendered page, fixed in the adapter's reason string.)
6. ⛔ **A run that lost its aggregation still reports everything else.** Reproduced on the E2 inputs by
   deleting the three `extended_pdm_score_*` rows (what navhard CV did on 2026-09-19): the headline
   reads **UNAVAILABLE — "the devkit wrote no `extended_pdm_score_combined` row … the per-token rows
   exist"**, while sub-metrics, zero rates, per-log, per-speed-band and the paired counts all render
   from the per-token rows. Pinned by `test_benchreport_contract.py`.
7. ⭐ **W2's two settled findings are rendered where they bind, not in a footnote.** (a) Every arm whose
   declared inputs contain `driving_command` is marked **⚠ command = route-level ORACLE** in a KPI tile
   beside its headline, tagged on its own chart row, named in a banner UNDER the headline chart carrying
   W2's evidence (reproduced 1,902/1,902 with OpenScene's own function; route = the expert's driven path
   at roadblock granularity, 79.8 % of forward blocks driven vs a 34.6 % null control; stage-2 command and
   route COPIED from the expert's frame, 5,462/5,462 navhard), and repeated as **NOT A ROUTE CLAIM** on
   the STRATEGIC family — whose nav-compliance readout is exactly the number that could be misread. On
   the E2 fixture the caveat fires on **A1, A1NT, A4**, and it is carried in the DATA
   (`arms.<arm>.caveats[]`), not only in renderer prose. (b) **No interval cell is ever blank**: each arm
   prints either `[lo, hi] · estimator · clusters · n` or UNAVAILABLE with its reason — for warmup, W2's
   settled rule verbatim (log-cluster bootstrap, clusters = `log_name`, B = 2000, n ≥ 8; warmup has 7),
   including the pre-CSV-frame case (`weight`, `log_name` dropped at `run_pdm_score.py:422`). (c) **the
   cross-track qualifier**: `cross_mae_m` is a LATERAL OFFSET at matched time index, not a distance to the
   path, so it cannot separate two plans that both keep y ≈ 0. The panel now prints W2's qualifier beside
   the LATERAL headline, the along-track context number that says whether the offset is informative (from
   the run's `_along_mae_m_for_context`, else the LONGITUDINAL family's `along_mae_m`, labelled as such),
   and names the projection-based alternative `headline.pathgeom_crosstrack_m`. **MEASURED on W1's real
   warmup run: CV and STOP both read 1.0658 m — the page now NAMES that tie** ("a moving arm and a standing
   one tie here by construction"). ⚠️ Fixing this exposed a defect of mine: W1's families are a THIRD
   layout (`families.<fam>.metrics.<k>`), which my curated lookup had been rendering as "—" — real values
   shown as blanks. The reader now normalises all three, and a refusal-shaped term (W2's stationary-plan
   `{status, reason, n, n_steps_total, min_ds_m}`) prints UNAVAILABLE with its reason, never a blank or a
   zero (real-run verify went 178 → 193 numbers once the blanks became values). (d) **`DRY_RUN_PASSED`
   is rendered as a PASS**, not a refusal: the banner reads "DRY RUN PASSED — the preflight succeeded and
   nothing was scored; every headline below is UNAVAILABLE by design, not by failure", and the mandatory-
   floor check drops from BLOCKING to a note (W1/E1: a passing dry run used to report REFUSED, a failure
   word on a success). (e) ⚠️ **A unit guessed from nothing, found on W1's new `internal_t1` run**: my axis
   read **"ADE (fraction)"** for a metric that is METRES and LOWER-is-better. The label now comes from the
   protocol's own metadata (`unit`), else names the column, and appends "· LOWER is better"; the paired
   section states that a negative Δ is then an improvement. Same class as the `df`/`step_s` scope traps —
   a true-looking label attached to the wrong quantity.
8. ⭐ **It renders W1's REAL run directories, not only the fixture — and the first one crashed it.**
   MEASURED 2026-09-20: `KeyError: 'wins'` on W1's first run, because on a two-stage protocol the
   per-scene win/tie/loss counts live **per stage** (the stages have different token sets); W1 then added
   pooled counts + `_wtl_scope`. The reader now normalises **three real layouts** (W1 per-stage, W1
   pooled + `_wtl_scope`, W5 fixture) in ONE module (`contract.py`), returning `(value, pointer)` so
   `data-k` never claims a key the run lacks; the page draws **one W/T/L chart per scope, each stating
   the scope beside the counts**, and a pooled figure always carries W1's own `_wtl_scope` sentence. All
   real run directories now render and verify PASS — including the **FAILED** run (8 numbers, the rest
   refusals) and the pre-fix run that has only per-stage counts (152 numbers). ⚠️ Two further defects
   surfaced here, both caught by the verifier and not by reading: a floor with no countable scope
   **vanished** from the page (now rendered as a named refusal), and the coverage rule demanded count
   keys a run need not have (now: *every count the summary HOLDS must be printed*).
9. ⚠️ **Two defects found in my own work by the checks, not by review.** (a) The refusal pointers for
   E1's `HUMAN`/`STOP` arms did not resolve in `summary.json` — the verifier caught 4. (b) A module
   named `html.py` in this package **shadowed the stdlib `html` package** for the NavSim-venv renderer;
   matplotlib's `import html.entities` died with a message naming neither file. Renamed to `page.py`,
   the NavSim-side script loads its helper by path and never touches `sys.path`, both pinned by a test.

### 1b. Retraction (banked in the same turn)

⛔ **NOT REPRODUCED — withdrawn:** my integration ask that `taniteval/taniteval/leaderboard/build.py:66`
imports `taniteval.report`. W4 checked all six of its modules and found none; re-checked here at the tip
(worktree and index): **0 occurrences**. The function I read on 2026-09-19 ~16:05 local (`_w5_charts`, with
`from taniteval.report import leaderboard_charts as w5` and the docstring *"…Unknown API → no charts."*) no
longer exists anywhere — `grep -rn "Unknown API" taniteval/` returns **0 hits** — and `build.py` was written
again at 2026-09-20 09:42. **Class: a true observation of another stream's LIVE file, quoted later as a
standing fact** (`repo advances mid-session` / `a summary is not a path`), not a wrong-scope grep — the
absent docstring is the discriminator. ⇒ **a `file:line` into a sibling's in-flight code is perishable and
must be re-read at write time, or cited with its timestamp.** The real defect there — a leftover fallback to
`taniteval.benchreport.html`, the module renamed to `page.py` mid-stream — was found and removed by W4, and
independently confirms the stdlib-shadow finding. W4's current code imports
`benchreport.leaderboard_charts.render_leaderboard_charts`, `benchreport.charts.{hbar_chart, nice_domain}`
and `benchreport.page.page`: those are now W4-facing API and W5 will not rename them unannounced.

## 2. What the report contains (the SPEC's acceptance list, one line each)

| # | content | where | state |
|---|---|---|---|
| H | protocol · devkit SHA + patch count · tier · loop stamps · closed-loop NO · evidence class · run id · FIXTURE flag, on screen **and on every printed page** | sticky/`position: fixed` header | ✅ verified on **29/29** PDF pages (`pypdf` text scan) |
| a | headline per arm vs the MANDATORY floors (STOP, CV, ECHO) as labelled reference lines + published rows of the **same protocol only** | §1, `fig/headline_official.svg`, `fig/headline_s2u.svg` | ✅ 1 published row drawn (warmup); a foreign-protocol row makes the verifier FAIL (tested) |
| b | sub-metrics per stage (NC DAC DDC TLC EP TTC LK HC EC) + multiplicative-zero rates + the devkit's own summary rows | §3 heat tables | ✅ stage-1 refusals grouped with reason + n |
| c | per-log and per-speed-band panels; paired W/T/L vs every floor | §2/§4/§5 | ✅ dot plots capped at 3 series (the validated all-pairs cap) + table twins with Δ |
| d | four-family panels, refusals with reason and n | §6 | ✅ every arm × family present; A1 STRATEGIC **PARTIAL** (nav-compliance 0.926, Δ vs nav-withheld +0.010) |
| e | failure gallery: BEV on the nuPlan map + agent boxes, plan vs CV vs STOP (human n/a with its reason), stitched camera with the plan projected, text overlay | §7 | ✅ 8/8 |
| f | leaderboard components for W4 (rank bars per protocol, params-vs-score scatter, our rows highlighted, floors as reference lines) | `leaderboard_charts.py`, `raw/leaderboard_components_demo.html` | ✅ (scatter: no numeric params in the source data — ask 5) |
| g | **W2 (a)** the route-oracle caveat beside every command-fed arm's headline + on the STRATEGIC family | §1 KPI tile, chart tag, banner; §6 | ✅ fires on A1 / A1NT / A4; carried in `arms.<arm>.caveats[]` |
| h | **W2 (b)** an interval or its reason for EVERY arm — never an empty cell | §1 | ✅ log-cluster bootstrap named; warmup refused at 7 < 8 clusters |
| k | **W1** `DRY_RUN_PASSED` rendered as a PASS that scored nothing; metric direction and unit taken from the protocol, never guessed | §0 banner, §1 axis, §2 lede | ✅ dry run: 0 critical issues, no "NOT MEASURED"; internal_t1 axis reads `ADE (column ade_dense_m) · LOWER is better` |
| j | **W2 (c)** cross-track qualifier + along-track context + the projection alternative; ties between arms NAMED; refusal-shaped lateral terms printed with their reason | §6 LATERAL | ✅ CV/STOP tie at 1.0658 m named on W1's run |
| i | **W1 real runs**: three count layouts read, per-stage counts rendered per stage, pooled counts labelled with `_wtl_scope` | §2, `contract.py` normaliser | ✅ every run dir under `taniteval/results/bench/navsim_v2/warmup_two_stage/` renders + verifies |

**Form** (the `dataviz` skill): its validated reference palette, validator run in BOTH modes and banked
(`raw/palette_validation.txt`); bars ≤ 24 px with 4 px data-ends; 2 px surface gaps; hairline solid
grid; legends for ≥ 2 series; direct labels + a **table twin on every chart** (the relief the light-mode
aqua's contrast WARN obliges); per-mark tooltips on hover AND keyboard focus; light and dark both
SELECTED (not flipped) and verified by rendering; print forced to the light tokens with figures kept
whole. Self-contained: inline CSS + inline JS, **no CDN, no web font**, gallery rasters as files.

## 3. Controls (each had to read a known value)

| id | control | result |
|---|---|---|
| V1 | parse-back of the rendered HTML vs `summary.json` | **PASS** 940 numbers / 46 refusals / 0 errors / 0 coverage gaps |
| V2 | M1 swapped arm label must go RED | **RED**, 230 errors |
| V3 | M2 value from the wrong arm must go RED | **RED** (`data-v` + printed text) |
| V4 | M3 CSV cell +1e-9 must be refused by the adapter cross-check | **REFUSED** (`cross-check FAILED`) |
| V5 | adapter output vs W1 schema v1 (`taniteval.bench.schema_check`) | **VALID** ×4 (E2 + E1, `summary` + `bench_run`) |
| V6 | boresight identity (analytic) | u, v = exactly ((W−1)/2, (H−1)/2) |
| V7 | projection round trip vs the frame bank's own ray model | **6.3e-6 px** (the reference is float32); the same identity in float64 **2.8e-14 px** |
| V8 | pinhole formula as a deliberate regression | **> 5 px** disagreement — the trap is detectable |
| V9 | mis-ordered basis on an OFF-AXIS point | **> 50 px** in u and v (⚠️ on-axis it is invariant — stated in the test) |
| V10 | legacy importers of `taniteval.report` before / after W5 | **27 / 27 passed** (`tests/legacy_importers_before_after.txt`) |
| V11 | `import taniteval.benchreport` does not pull in torch | **PASS** |
| V13 | the oracle caveat sits BETWEEN the headline and the paired section (not a footnote), names every command-fed arm, and is present in the summary data | **PASS** (`test_benchreport_contract.py`) |
| V14 | no arm's interval cell is blank; an OK interval renders with estimator + cluster unit + n | **PASS** (synthetic navhard-shaped interval: `[0.1712, 0.1994] log_cluster_bootstrap · clusters log_name · n 76`) |
| V15 | every real W1 run directory renders and verifies (parametrised at collection, so it tracks W1's live results dir) | **PASS** on each existing run, incl. the FAILED one |
| V16 | all three count layouts (pooled-only, per-stage-only, both) render with their scope; a counts-free paired block is refused, not dropped | **PASS** (`test_benchreport_w1runs.py`) |
| V17 | the normaliser reads both `per_stage` spellings (`score` / `scene_mean`) and returns the REAL pointer | **PASS** |
| V18 | over every real run that has a lateral block: the values render (not blanks), the qualifier + alternative are present, and a tie is named | **PASS** (parametrised over W1's runs) |
| V19 | W2's new annotations (`_cross_is`, `_along_mae_m_for_context`, `_projection_based_alternative`) and the refusal shape render; a refusal never prints as a blank or a zero | **PASS** (`test_benchreport_w1runs.py`) |
| V20 | a `DRY_RUN_PASSED` run renders as a pass: no critical issue, no "NOT MEASURED"/"BLOCKING", verify PASS | **PASS** |
| V21 | a lower-is-better protocol says so and is never called a "fraction" | **PASS** (pins the real internal_t1 defect) |
| V12 | full W5 + legacy suite | **81 passed** (`tests/pytest_benchreport.txt`) |

## 4. The showcase

`raw/showcase_e2_warmup/` — a §1 run directory adapted from E2's banked outputs (`bench_run.json`,
`summary.json`, `scores/`, `artifacts/`, `criteria/`, `plans/`, `scenes.json`) plus the rendered
`report/` (index.html, 7 SVG figures, 8 gallery PNGs, `verify.json`). **4.08 MB total, figures
2.87 MB** (the `du` figure of 53 MB is exFAT's 1 MiB clusters, not bytes). It is a FIXTURE:
`provenance.fixture = true`, `leaderboard_eligible = false`.

Served for review at **http://127.0.0.1:18777** (local, `python -m http.server`, background task
`bfjztr7bm`), root = `raw/showcase_e2_warmup/report`.

## 5. Scope and limits (named, not hidden)

* the gallery renders **stage-2 synthetic scenes**; a stage-1 scene needs the log pickles (W1/W3 input)
  and is refused by name;
* **human** is n/a on every warmup stage-2 scene by protocol (`num_future_frames = 0`, E1 MEASURED
  204/204) — printed on each figure rather than left blank;
* the stitch is **rotation-only**, so side-camera regions carry their own parallax; the plan sits in the
  front cone (stated in `camproj.py`);
* published rows are only as good as W4's file; the report prints its path and evidence class;
* `nuscenes_ol` / `internal_t1` runs fall back to the generic path (headline + families + provenance)
  and say so — protocol-specific panels land with W6/W3's first run;
* the W5 verifier checks what the page prints against `summary.json`; it does **not** re-derive
  `summary.json` from the CSVs (that is the adapter's cross-check, V4) or re-run the devkit;
* W1's results directory is LIVE (runs appeared and disappeared while this package was being written), so
  the real-run test globs `results/bench/*/*/*/summary.json` at collection time — every benchmark, no
  pinned run ids;
* ⚠️ **W5 wrote inside W1's runs of record**: rendering a real run with the CLI writes `report/index.html`,
  `report/fig/*.svg` and `report/verify.json` into that run directory (that is the contract — the report
  lives in the run). Those files are W1's to stage; W1's end-of-turn check caught them, which is the rule
  working. The TESTS never write there (they render into a tmp dir via `out_dir`);
* ⚠️ **provenance note**: the session scratchpad is shared between streams (W2, 2026-09-20 — a same-named
  script was overwritten by another package). Everything provenance-bearing here was re-run on 2026-09-20
  writing DIRECTLY into this package (`raw/palette_validation.txt`, `tests/legacy_importers_before_after.txt`,
  `tests/pytest_benchreport.txt`) or under `w5_`-namespaced scratch files (the print-page check).

## 6. Deliverable manifest (staged, NEVER committed — `git add` by exact path only)

| artifact | where it lives | staged |
|---|---|---|
| the report package (16 modules: `__init__.py`, `__main__.py`, `adapt.py`, `camproj.py`, `charts.py`, `cli.py`, `contract.py`, `gallery.py`, `leaderboard_charts.py`, `metrics.py`, `navsim_gallery.py`, `page.py`, `palette.py`, `render.py`, `svg.py`, `verify.py`) | `taniteval/taniteval/benchreport/` | ✅ verified (index blob == worktree, both 40 chars) |
| tests (5) | `taniteval/tests/test_benchreport_contract.py`, `taniteval/tests/test_benchreport_golden.py`, `taniteval/tests/test_benchreport_legacy_compat.py`, `taniteval/tests/test_benchreport_projection.py`, `taniteval/tests/test_benchreport_w1runs.py` | ✅ verified |
| SPEC (pre-registered + amendment A1) · PLAN · RESULT · COMMS | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-visual-reporting/` | ✅ verified |
| pre-registration hashes · palette-validator output (both modes, re-run into the package) · W4 chart-components demo | `…/raw/{SPEC_PREREG_HASH.txt, palette_validation.txt, leaderboard_components_demo.html}` | ✅ verified |
| **the showcase**: a §1 run directory + its rendered report (56 files, 4.17 MB; figures 2.95 MB) | `…/raw/showcase_e2_warmup/` (`bench_run.json`, `summary.json`, `scores/`, `artifacts/`, `criteria/`, `plans/`, `scenes.json`, `report/index.html`, `report/fig/*.svg`, `report/fig/gallery/*.png`, `report/verify.json`) | ✅ verified |
| test evidence: full-suite output · legacy-importer verification (re-run into the package) | `…/tests/{pytest_benchreport.txt, legacy_importers_before_after.txt}` | ✅ verified |
| ⛔ NOT touched | `taniteval/taniteval/report.py` (blob `d5633f68` == HEAD, `git status` clean), W1's `bench/**` and its run directories (rendered into a tmp dir by the test), W4's `leaderboard/**`, `published_results.json` | — |
| live for review (not an artifact) | `http://127.0.0.1:18777` → the showcase report (background task `bfjztr7bm`, `python -m http.server`, root = `raw/showcase_e2_warmup/report`) | — |
