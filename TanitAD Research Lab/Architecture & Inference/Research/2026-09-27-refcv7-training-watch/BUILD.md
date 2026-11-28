# refcv7 Training Watch — BUILD record (2026-09-27, ~05:10 Berlin)

**Owner:** the Training Watch builder (subagent), for the Master Mind (the single committer). Nothing staged, committed or pushed.
**Base:** tip `03e6ed9a2982339cb03114e7857995e5eab6c54b` (SPEC A11). Both repo files are NEW at that tip. Everything below was re-run on a clean `git archive` of `03e6ed9` plus the two files.
**Lands via:** `LANDING_READY.txt` in this package.

## Integration

1. **The launch gate's G-MAP item 4** builds this Watch once `PROFILES["refcv7"]["map_hires"]["watch_builder"]` is set in the gate package's `launch_gate.py`. The Master Mind has routed it to the gate agent; this package does not touch the gate. The template, tested exactly as `launch_gate.py` formats and runs it (`cwd=tree`, rc 0; the gate's OWN `map_watch_reasons` returned `[]` and found the alarm tile):
   ```
   ["{python}", "{tree}/taniteval/tools/training_watch/build_watch_refcv7.py", "--metrics", "{metrics}", "--out", "{out}"]
   ```
2. **The run directory** is read from `--out` of `stack/ops/runs.d/refcv7-r101-s0.argv.json` (the file `sup_refcv7.sh` runs). Until that file lands, pass `--run /home/nvidia/refcv7_run/runs/refcv7-r101-s0` or set `$REFCV7_RUN`.

## The Master Mind's rulings (2026-09-27), as applied

| # | ruling | how the page implements it |
|---|---|---|
| 1 | **Thin-class alarm = the REGISTERED rule only** (LOGGING_SPEC_MAP10 5.3) | RED iff any class's `0_20` eval IoU ≤ 0.05 at any eval at or after step 5,000, OR any eval row lacks one of the 40 keys. **Latched.** The tile shows "RED (latched) · first red at step N", what first tripped it, and the current value ("now (step X): lowest 0–20 m IoU <class> <v>"). **20–60 m ≤ 0.05 from step 5,000 = AMBER, informative** (an amber outline and a `~` marker, an amber tile and chip when nothing is red). **Beyond 60 m: values only** (no fill, no outline, no status, whatever the value). The rule reads the IoU as logged, with no ground-truth condition. A null IoU (absent from GT and every prediction) is no value, so it is neither a breach nor a missing key; it prints "undefined". The page says, in the map section and on the tile, that the per-band bar against refcv6@38k (A8) needs the paired interval and is read by the battery at milestones, never by this page |
| 2 | **Box ratio alarm armed from step 5,000** | Before it the tile is grey, "warming up (presence prior 0.01)", with the value (or UNAVAILABLE) shown, and the chip is neutral. From step 5,000: outside [0.5, 1.5], absent or null is an ALARM. The trainer's own `conf_ratio_alarm` flag is held against this page's BAND reading, not the armed state, so a warm-up ratio outside the band is not reported as a disagreement |
| 3 | **EvalFlyWheel: arm `R7_A1`, package `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-27-refcv7-standard-tests/`** | The defaults are `R7_A1` and `D:\Projects\TanitAD\FlyWheels\TanitAD_EvalFlyWheel\incoming\2026-09-27-refcv7-standard-tests` (env overrides `$REFCV7_NAVSIM_ARM` / `$REFCV7_EVAL_PKG`). Until the package exists (first milestone) the section reads "No banked NavSim milestone was readable … nothing is shown rather than a guess". No live lane is named, so an unbanked split reads **"not banked"**, never "not run" |
| 4 | **Pace chip: MARGINAL pace vs 8.0, amber, labelled** | The chip reads "pace X s/step (marginal, incl. eval + ckpt) ≤/> 8.0 · warm median Y", amber above the line and never red. The warm median (intervals without an eval or checkpoint) sits beside it. The page says the PI rule itself is applied at the G-LIVE smoke, not by the Watch |

## What is built

`taniteval/tools/training_watch/build_watch_refcv7.py` (standard library only; compiles on Python 3.9, 3.12 and 3.13; Thor's `tanitad-train` is 3.12.3, read-only probe) and `taniteval/tests/test_build_watch_refcv7.py` (45 tests).

**Kept from refcv6** (`build_watch_refcv6.py`, unchanged on the tip): the Windows OpenSSH pull; the stderr size AND content read in one breath, every line diagnosed or it fails the chip; the whole-token supervisor parse; elapsed_s-reset segments, planned vs unplanned; the chips; the charts and hover; the ETA; the NavSim count guard with `UNAVAILABLE — count guard FAIL (n of N)`; "What it says, and what it does not"; Berlin times (EU DST rule, no tz database); the T0 stamp.

**Parameterised:** `--arm` / `$REFCV7_ARM` (default `refcv7-r101-s0`); the run dir; `--dir` / `$REFCV7_WATCH_DIR` (default `C:\Users\Admin\qland\work\watch7`); `--out`, `--summary`. TOTAL = 50,400 (SPEC §4), or the run's own `config.json` `--steps` if it differs (flagged).

**`--metrics <file> --out <page>`** builds from ONE metrics.jsonl (config.json beside it when present) and never pulls; every other missing artifact renders UNAVAILABLE. That is the gate's form. **Outputs:** the page, `<page stem>.summary.json`, and a `ZZWATCH {…}` line on stdout.

### What refcv7 adds

| section | what it shows | its alarm |
|---|---|---|
| **Map (10 cm)** | class × band heatmap of the latest eval (declared rule; far bands plain), n printed; the RAW-argmax heatmap; the loss-share table; 8 per-class charts × 5 bands = all 40 series, each carrying its key; the 10 cm loss; the per-class training signal (loss share vs label-mass share, and the pull `gno` on each class's own cells) | the **THIN-CLASS ALARM** (tile `id="thin-class-alarm"`), ruling 1. **MAP SIGNAL** chip: a class with labelled cells and exactly 0 pull in the latest train row (LOGGING_SPEC_MAP10 5.2). The IoU is re-derived from each row's own inter / union counts; with no counts the page says the check could not run |
| **Boxes (P0)** | per head: class-mean AP at 0.5/1/2/4 m × all/0–20/20–40/40–60 m; AP@2 m; presence AUROC; P/R/F1 at σ ≥ 0.5; confident / VIS-1 positives with its census; class accuracy on TPs; centre error; the INFORMATIVE TRAIN P = R gate; trend charts; refcv6@38k reference lines | **BOX RATIO ALARM** per head, ruling 2. Cross-checks: the ratio against n_conf / n_pos, P against tp / n_conf, R against tp / n_pos, the trainer's flag against the band reading |
| **Gradient reach (D3)** | every declared `ga_*` group: latest value, params with grad, min over the run, rows dead or absent, a log10 sparkline; batch 3's `grad_unreachable` as information | any DECLARED key absent or exactly 0 in the latest train row; no declaration = UNAVAILABLE and an alarm |
| **Residual prior (NEW-1)** | `config.json` `seams.residual_prior` against argv; any logged `residual`/`prior` key | a chip if argv and stamp disagree. MEASURED from source at the tip: no per-row prior key is logged |
| **Pace** | ruling 4; a pace chart with refcv6's 6.4 and the 8.0 line; ETA; checkpoint cadence vs SPEC's 500 | amber above 8.0, never red |

A key the page needs and the log lacks renders **UNAVAILABLE** and raises that section's alarm; it is never drawn as 0. ⛔ **A key absent from EVERY row prints no key name anywhere**, because the gate reads the key literals as "this series is carried".

## Key names consumed

| keys | status | source |
|---|---|---|
| `eval_map_hires_iou_{cls}_{band}` (40), `…_iouraw_*`, `…_lshare_*`; cls ∈ nocls, drivable, lane, crosswalk, arrow, edge, hatched, sidewalk; band ∈ 0_20, 20_40, 40_60, 60_80, 80_100 | **VERIFIED as the landed contract** (SPEC_REFCV7 A8 item 3); the emitter (NEW-2) is not landed | SPEC A8; NEW-2 BUILD.md sec. 4 |
| `eval_map_hires_{n,inter,union}_*`, train `map_hires`, `map_hires_{lshare,n,gno}_*`, `eval_map_hires`, `config.json` `map_hires.decision_rule` | VERIFIED as names (LOGGING_SPEC_MAP10 sec. 2-3, landed), 5 bands per A8 | same |
| `grad_reach_logging.{declared,keys}`, `declared_vs_built.grad_unreachable`, `seams.residual_prior` | **VERIFIED** on the tip (`refc_v3_train.py`; batch 3 `4797ffb` for `grad_unreachable`) | tip |
| `eval_{h}_prec@gate`, `rec@gate`, `conf_ratio`, `ap2m`, `auroc_matched`, `auroc_objectness`, `cls_acc_tp`, `centre_err_p50` | names VERIFIED in the landed `LOGGING_SPEC_BOX.md`; emitter not landed | box-head audit |
| `eval_{h}_f1@gate`, `conf_ratio_alarm`, `n_conf`, `n_pos`, `tp@gate`, `n_ignore`, `n_windows`, `cls_acc_tp_priorcorr`, `det_map{0p5,1,2,4}_{all,0_20,20_40,40_60}`, `calib_{pr_gate,prec,rec,n_pos,n_windows}`, `eval_calib_error` | ⚠️ **UNVERIFIED** | the box builder's IN-PROGRESS `detection_metrics.py` (`C:/Users/Admin/bxh_work/overlay/…`, read 03:34); the D: package did not exist when checked |
| supervisor: `ZZ<arm>-<step>-<steps>-<n_err>-<launch>ZZ`, `ZZGATEOK-…`, `ZZGATEREFUSED-…`, `lock acquired … supervisor pid N`, `launch #N of M:` | ⚠️ UNVERIFIED until the gate lands | the gate package's `sup_refcv7.sh` |
| NavSim arm `R7_A1`, package path | **set by the Master Mind's ruling**; the package itself does not exist yet | ruling 3 |

refcv6@38k reference lines (MEASURED, the landed box-head audit `RESULT.md`, 556 clipgrid eval windows, gate 0.5, vs the trainer's targets): AP@2 m 0.210 (VIS-1 0.151); P 0.162 / R 0.551 (VIS-1 0.109 / 0.578); ratio 3.40 box3d / 3.75 agent; AUROC 0.855 / 0.860. Labelled as other windows, not a paired comparison.

## A note for the key contract (INHERITED from the Master Mind, 2026-09-27, not re-verified here)

The early G-MAP-OVERFIT MAIN arm failed **lane (0.417)** and **edge (0.076)** at step 1,000. A lift lever may change NEW-2's logged keys. This builder reads the 40-key contract as SPEC A8 registers it; if the contract changes, the Master Mind says so and the `MAP_CLASSES` / `MAP_BANDS` literals (and the tests' own literals) change with it.

## Tests and red arms (`raw/`)

- **62 passed, 0 failed** on the clean `03e6ed9` tree plus the two files: `test_build_watch_refcv7.py` (45) and `test_training_watch_refcv6.py` (17), `taniteval` asserted inside that tree (`raw/pytest_03e6ed9_final.log`). Expected values are literals; the 8 classes and 5 bands are written out from SPEC A8, not imported.
- **Map:** one key missing (UNAVAILABLE + alarm, never 0); a key missing from every row (no literal on the page); 0–20 m at 0.03 from step 5,000 → RED, "first red at step 5,500", the current value; the step-5,000 start (control at 4,500); the latch (0.05 at 5,000, recovered at 5,500 → still RED with "first red at step 5,000" and "now … above 0.05"); any class at 0–20 m is RED while 20–60 m is amber; 20–60 m at or below 0.05 → AMBER, not red; beyond 60 m → values only, an undefined far cell plain and never 0; the literal rule with no ground-truth condition; an IoU that does not reproduce from its counts; the training signal (cells and no pull → alarm; no cells → no evidence).
- **Boxes:** 3.4 → alarm, 1.0 → none, edges 0.5 / 1.5 inside, 0.49 → alarm; warming up before step 5,000 (grey, value shown, no false flag disagreement) and armed at 5,500; a ratio that does not reproduce from the counts; absent P0 keys.
- **Gradient reach:** declared `ga_mh_refine` absent → alarm, exactly 0 → alarm, an undeclared zero → none; no declaration → UNAVAILABLE.
- **Pace:** 8.5 → amber with the ruling's label, 6.5 → none. **Gate form** from ONE metrics file; a drivable-only log fails the gate's reading. **NavSim:** banked KPIs, partial navtest refused, wrong stage counts refused, a wrong arm id refused, "not banked" without a live lane, the ruling's defaults. **refcv6 health checks** kept; the Berlin DST rule; the run dir read from the gated argv; an unsafe run dir refused.
- **Mutation proof** (`code/mutation_proof.py` → `raw/mutation_proof_training_watch_refcv7.json`): **17 of 17 CAUGHT**, control 45/0 before and after, on the `03e6ed9` tree. M1 absent key drawn as 0 · M2 null drawn as 0 · M3 amber promoted to RED · M4 no start step · M5 latch lost · M6 a missing key prints its literal · M7 A10 band widened · M8 ratio cross-check removed · M9 / M10 absent / zero declared ga not an alarm · M11 count guard off · M12 the IoU cross-check reads agreement when it checked nothing · M13 zero pull on cells not an alarm · M14 box alarm armed from the first eval · M15 far bands coloured · M16 a ground-truth condition added to the registered rule · M17 the trainer's flag held against the armed state.

## Renders (`raw/smoke/`, from `code/smoke_render.py`, on the `03e6ed9` tree)

- `FIXTURE_synthetic_healthy.html`; `FIXTURE_synthetic_alarms_on.html` (lane 0–20 m 0.03 → RED first red at 5,500; edge 20–40 m 0.03 → amber; arrow 80–100 m 0.0 → a plain value; box3d 3.4 armed → ALARM; `ga_mh_refine` 0); `FIXTURE_gate_form_watch_refcv7.html`: synthetic, NOT a run.
- `BACKCOMPAT_refcv6-r101-s0_log_by_the_refcv7_builder.html`: the REAL refcv6-r101-s0 artifacts (a copy of `C:/Users/Admin/qland/work/watch6/`), with refcv6's own segments and diagnosed stderr line. All 11 shared charts drawn; step 38,250, 3 segments, 0 unplanned, stopped by the PI; map / boxes / grad reach / prior UNAVAILABLE.
- **The gate's own reader** (`launch_gate.map_watch_reasons`, `PROFILES["refcv7"]`, imported read-only): both fixtures and the gate form → `[]` with the alarm tile found; the refcv6 page → FAIL (24 series missing), as it must.
- Looked at in a browser earlier (desktop and 375 px, before the rulings): no page-level horizontal scroll; the heatmap scrolls inside its wrapper. After the rulings, the alarm tile, the cell classes (`breach` / `amber` / `plain`) and the pace chip were read back from the rendered HTML.

## Process note (fail loud)

The binding RAM start gate (≥ 7.5 GB free on 3 samples) was not applied as written to the early runs: they were checked on ONE printed sample (7.6–9.3 GB), and one of them, the second mutation proof, started at **6.62 GB free** because the command printed the sample instead of gating on it. It was a ~100 MB pytest loop and finished cleanly. Every later run, including every run for the rulings and the final chain, went through a real 3-sample gate (`ramgate.ps1`, scratchpad).

## Deliverable manifest

| artifact | where | only one place? |
|---|---|---|
| `code/fix/taniteval/tools/training_watch/build_watch_refcv7.py` (blob `86e47b33`, LF) | this package (D:) + the test tree `C:/Users/Admin/w7tree_03e6ed9` | no (two, blob-equal) |
| `code/fix/taniteval/tests/test_build_watch_refcv7.py` (blob `b3041189`, LF) | this package + `w7tree_03e6ed9` | no (two, blob-equal) |
| `code/mutation_proof.py`, `code/smoke_render.py` | this package | yes, until landed |
| `raw/mutation_proof_training_watch_refcv7.json`, `raw/mutation_proof.log`, `raw/pytest_03e6ed9_final.log` | this package | yes |
| `raw/smoke/*` (4 pages, 4 summaries, `smoke_record.json`, `smoke_render.log`) | this package | yes |
| `BUILD.md`, `LANDING_READY.txt` | this package | yes |
| scratch: `C:/Users/Admin/w7tree_a3db3a8` (authoring tree), `w7tree_c8094e6`, `w7tree_03e6ed9` (the shipped combination) | C: | disposable |
