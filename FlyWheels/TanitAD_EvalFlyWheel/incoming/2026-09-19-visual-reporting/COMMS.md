# COMMS — W5 (visual reporting)

## Decisions MADE in this stream (each reversible; the orchestrator / PI may overrule)

| # | decision | why (evidence) |
|---|---|---|
| D1 | The package is **`taniteval/taniteval/benchreport/`**, CLI `python -m taniteval.benchreport <run_dir>`; the legacy `taniteval/taniteval/report.py` is **untouched** (blob `d5633f68` == HEAD). | Orchestrator ruling 2026-09-19. A `report/` package shadows the module for three live importers (`runner.py:641`, `rerun_all.sh:17`, `tests/test_estimator_closeout.py:30`). Before/after counts banked: **27 / 27 passed** (`tests/legacy_importers_before_after.txt`), pinned by `test_benchreport_legacy_compat.py`. |
| D2 | The report's **headline is the PROTOCOL's official statistic** (here the devkit's two-stage combined EPDMS). E2's pre-registered **S2-EPDMS-u** is a SECOND axis, labelled as a stage-2-only statistic. | `navsim.cross_protocol` + the ~30-point trap the registry records: two statistics wearing one name must never share an axis. Camera arms' official row is HYBRID ⇒ their headline is `UNAVAILABLE` with the reason, and **the hybrid number is never printed** (E2 RESULT §2). |
| D3 | The E1/E2 **fixture adapters** emit W1 schema v1 (`taniteval.bench.summary/1`) with `provenance.fixture = true`, `leaderboard_eligible = false`. | W1's producer is not written yet; the report had to be built and tested against REAL outputs. W4 must not double-count them against W1's canonical run of the same protocol. |
| D4 | Gallery **selection rule**: stage-2 scenes the arm answered ITSELF, ranked by score ascending, then by the arm's deficit against the best floor, then by token; ≤ 2 per log. Printed in the report. | A gallery that silently picks scenes is an argument nobody can check. The deficit tie-break surfaces the failures where standing still or CV was fine. |
| D5 | The camera projection is the **exact inverse of the frame bank's own ray model** (`tanitad.data.calib.cylindrical_rays`), from **cam_f0's optical centre**, ego frame == lidar frame. | The stitch is rotation-only (`build_navsim_eval.py::build_map`), OpenScene's `lidar2ego` is the identity (MEASURED in the log metadata), CAM_F0 sits at z = 1.52 m ⇒ ground plane is z = 0. Round trip closes to **2.8e-14 px** in float64 and **6.3e-6 px** against the bank's float32 rays; the pinhole formula (the CLAUDE.md trap: 92.6° vs 120°) is a RED regression arm. |
| D9 | **Rendering a real run writes into that run's directory** (`report/index.html`, `report/fig/*`, `report/verify.json`) — that is the §1 contract, so W5's CLI touches W1-owned directories by design; the TESTS never do (they render into a tmp `out_dir`). W5 stages none of it: those files belong to the run's owner. | An ownership boundary is not a lock (orchestrator, 2026-09-20): a MISMATCH on a file you did not touch is evidence about the TREE. W5 re-runs its blob comparison at the end of every turn and reports what moved. |
| D6 | Exit codes: **1** when the page's own numbers disagree with `summary.json`, **3** when a requested gallery could not be drawn (the page still renders, with a BLOCKING banner). | "Assert on the artifact": the report verifies its own HTML and writes `report/verify.json`. |
| D7 | **W2's two settled findings are rendered as first-class page elements** (orchestrator relay 2026-09-20): (a) the route-oracle caveat appears in a KPI tile beside the headline, as a tag on the arm's own chart row, as a banner under the headline chart, and as **NOT A ROUTE CLAIM** on the STRATEGIC family — it fires from `declared_inputs` AND is carried in `arms.<arm>.caveats[]`; (b) every arm prints an interval or its reason, never a blank cell, naming the log-cluster bootstrap (clusters `log_name`, B = 2000, n ≥ 8) and the pre-CSV-frame case (`run_pdm_score.py:422`); (c) the LATERAL panel prints the cross-track qualifier beside the headline, the along-track context number, the projection-based alternative (`headline.pathgeom_crosstrack_m`), and NAMES any tie between arms (CV and STOP both 1.0658 m on W1's real run) — plus refusal-shaped lateral terms with their reason; (d) W1's `DRY_RUN_PASSED` renders as a PASS that scored nothing (no BLOCKING, no "NOT MEASURED"), and the headline axis takes its unit and direction from the protocol instead of guessing "(fraction)" — the real `internal_t1` run exposed that one. | An oracle caveat in a footnote is the `true but wrong for the reader` class; a blank interval cell reads as "no uncertainty" rather than "no admissible estimator". Pinned by `test_benchreport_contract.py` (V13/V14). |
| D8 | **One reader, three layouts.** `contract.py` normalises W1's per-stage/pooled paired blocks, W1's `per_stage.<stage>.score` vs W5's `scene_mean`, W1's flat `submetrics.combined_row.<K>_s1/_s2`, and W1's `per_log.<log>.<stage>.score_mean` vs W5's `per_log.logs.<log>.value` — always returning the REAL pointer, so `data-k` never claims a key the run lacks. Per-stage counts are drawn per stage; a pooled count is never drawn without W1's `_wtl_scope` sentence beside it. | On a two-stage protocol the stages have different token sets, so pooled ≠ per-stage; printing one where the other is meant is the `true but wrong for the reader` class. Pinned by `test_benchreport_w1runs.py` (real runs + all three layouts + a counts-free block). |

## ⭐ Integration asks (escalated — also in the RESULT headline)

1. **W1 — emit two files the gallery needs** in every run directory, or it cannot be drawn:
   `plans/<arm>.npz` (`token`, `poses [N,8,3]`, `source`, `sampling [n_poses, dt]`) and `scenes.json`
   (`tokens{token: {stage, log, v0, command, scene_token, map_name, pickle, frame_type,
   num_future_frames}}` + `frame_bank`, `synthetic_scene_pickles`, `maps_root`).
   `taniteval/taniteval/benchreport/adapt.py` writes exactly this shape from E2's seam files — copy it.
2. **W1 — the inner layout of the schema's OPEN objects** (`per_stage`, `submetrics`, `per_log`,
   `paired`, `statistics` are all `{"type": "object"}`). W5 now reads BOTH spellings and needs no change
   from you, but the list is worth freezing in the schema so a third one does not appear:
   stage number `per_stage.<stage>.score` (W1) or `.scene_mean` (W5 adapter); sub-metrics
   `per_stage.<stage>.submetrics.<K>` (W1/W5) or the flat `submetrics.combined_row.<K>_s1/_s2` (W1);
   per log `per_log.<log>.<stage>.{n, score_mean}` (W1) or `per_log.logs.<log>.{n, value}` (W5);
   paired counts pooled at `paired.<floor>.{wins,ties,losses}` **with `_wtl_scope`** and/or per stage at
   `paired.<floor>.by_stage.<stage>.{wins,ties,losses,score_mean_delta,submetric_mean_deltas}`; statistic
   deltas `paired.<floor>.S2_EPDMS_u_delta` (W1) or `.statistic_deltas.<name>` (W5). Optional and read
   when present: `per_speed_band.bands.<id>.{label,n,value}`, `paired.<floor>.{by_log,by_speed_band}`.
   The reader is ONE module (`contract.py`).
3. **W1 — keep `_wtl_scope` and `by_stage` exactly as they are.** The report reads both and labels each
   scope; if the pooled counts ever lose `_wtl_scope`, the page falls back to "scope not declared by the
   run", which is honest but much weaker than your sentence.
4. **W1 — keep `headline` = the protocol's official statistic**; put any second statistic under
   `statistics`. The report draws them on separate axes and refuses to mix.
5. ⛔ **RETRACTED — NOT REPRODUCED.** I wrote that `taniteval/taniteval/leaderboard/build.py:66` imports
   `taniteval.report`. W4 checked all six of its modules and the string occurs in none; re-checked here at
   the current tip, **worktree AND index: 0 occurrences**. What I read on 2026-09-19 (~16:05 local) was a
   `_w5_charts()` whose body was `from taniteval.report import leaderboard_charts as w5`, docstring
   *"Import W5's chart components if they exist — never edit them. Unknown API → no charts."* — a function
   that no longer exists anywhere (`grep -rn "Unknown API" taniteval/` → **0 hits**), and `build.py`'s mtime
   is 2026-09-20 09:42, after my read. ⇒ **ROOT-CAUSE CLASS: an observation of ANOTHER STREAM'S LIVE FILE,
   quoted later as if it were still current** — the `repo advances mid-session` / `a summary is not a path`
   family. ⚠️ It was NOT a grep read at the wrong scope: the absence of the old docstring is the
   discriminator, and the scope was `taniteval/taniteval/leaderboard/` at the time. **The rule it earns:
   a `file:line` into a sibling stream's in-flight code is PERISHABLE — re-read it immediately before
   asserting it in a deliverable, or cite it as "at <time>", never as a standing fact.**
   ⭐ **The real defect in that area was found and removed by W4, not by me:** a leftover fallback to
   `taniteval.benchreport.html` — the module W5 renamed to `page.py` mid-stream. It independently confirms
   the stdlib-shadow finding (ask 8) and shows the rename was an INTERFACE change, not an internal tidy.
   ✅ **Current state, checked at the tip — W4 is already wired to W5, and four names are now W4-facing
   API that W5 will not rename without telling W4:** `benchreport.leaderboard_charts.render_leaderboard_charts(model)`
   (build.py:70), `benchreport.render_leaderboard_charts` (its second candidate),
   `benchreport.charts.{hbar_chart, nice_domain}` (render.py:593) and `benchreport.page.page` (render.py:685).
6. **W4 — numeric parameter counts.** No row in `published_results.json` carries one, so the
   params-vs-score scatter cannot be drawn for any protocol (only four v1 rows carry an `encoder`
   STRING like `"21M"`, which the component will parse but labels as an ENCODER count, never a total).
   Please add `params_total` / `params_encoder` with their sources.
7. **W2/W4 — keep `protocols[*].{unit, decimals, higher_is_better, metric, title}`** in
   `published_results.json`; the report reads them to pick the axis unit and precision.
8. **Suite-wide lesson (already fixed here, worth a rule):** no module inside a package that any
   script puts on `sys.path` may share a **top-level stdlib name**. MEASURED 2026-09-20: a
   `benchreport/html.py` shadowed the stdlib `html` package for the NavSim-venv renderer, and
   matplotlib's `import html.entities` (via pyparsing) died with *"attempted relative import with no
   known parent package"* — a failure that names neither the file nor the collision. Renamed to
   `page.py`; the NavSim-side script now loads its helper BY PATH and never touches `sys.path`; both
   pinned by `test_benchreport_legacy_compat.py`.
9. **E4 / leaderboard currency** — the showcase run directory under `raw/showcase_e2_warmup/` is a
   FIXTURE (`leaderboard_eligible: false`). If W4's globber ever reads `FlyWheels/**/summary.json`,
   honour that flag.
