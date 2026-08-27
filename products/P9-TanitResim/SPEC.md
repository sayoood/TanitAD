# P9 — TanitResim: inventory and consolidation spec

**Status:** DRAFT for PI review · **Owner:** EvalFlyWheel · **Written:** 2026-08-23
**Evidence class:** every row is **MEASURED** — the source file was read or grepped in
this worktree and the line number is cited — unless marked `INHERITED` or `UNVERIFIED`.
Commit dates are `git log -1 --format=%ai <branch> -- <path>` on `claude/zen-bose-eb35e9`.

> ⚠️ **§0 corrects the charter's file list before anything else uses it.**

---

## 0. Three corrections to the charter

**0.1 — `taniteval/tools/corpus_overlay.py` does not exist.** The file is at
`taniteval/taniteval/corpus_overlay.py`. Both paths were probed.

**0.2 — ⭐ TanitResim already exists as a complete, tested web application, and it is
not in `taniteval/` at all.** It is `stack/tanitad/resim/` (exporter + SPA), served by
`stack/scripts/resim_app.py`, fed by `stack/tanitad/replay/`, with **39 tests** in
`stack/tests/test_resim.py` and a serve-ready demo bundle checked in at
`TanitAD Research Lab/Tools&DevEnv/Implementation/incoming/2026-07-24-tanitresim-productionization/sample-bundle/`.
None of the charter's four starting points point at it.

**0.3 — There is a third renderer family the charter does not mention at all:** the
AlpaSim/NuRec closed-loop stack in `stack/experiments/alpasim-gsplat/`. Its
`overlay_video.py` is, on the evidence below, **the single most standard-compliant
renderer in the programme** — better than "THE STANDARD" itself on camera geometry and
on metric families.

**So P9 is not a green-field build and not a rescue of fragments.** It is the
consolidation of **three mature, independently-built renderer families** that implement
the same visual contract three times — in PIL, in JS/canvas, and in cv2 — from three
different data paths, with three different camera models. **That triplication is the
problem this spec exists to solve.** Around them sits a long tail of ~25 banked one-off
renderers (§2.5) which is a symptom, not a target.

---

## 1. What TanitResim is

**TanitResim** is TanitAD's product **P9**: the programme's replay-and-visualization
pipeline — the one tool, in a CLI mode and a UI mode, through which a trained arm's
behaviour is *looked at* rather than summarised. It consumes the eval harness's banked
artifacts (it never re-derives a scored quantity and never re-runs a scoring collector),
and it renders them under one binding visual contract: every trajectory view shows the
**camera projection**, a **metric BEV**, the decoded **tactical manoeuvre** and the
**strategic route/goal** *together*, with **ADE** — and when one of those is genuinely
unavailable it **says so on the frame, with a reason**, rather than silently dropping
it. Its CLI produces archival reels and stills for the record; its UI lets an operator
scrub an episode, compare arms side by side, and jump to where an arm failed. It is the
visual half of the eval story whose numeric half is P7 (TanitEval).

---

## 2. Inventory

### 2.1 Family A — the interactive stack (this *is* TanitResim)

| # | Path | Size | Last commit | What it is | Kind | Referenced by | Status | **Disposition** |
|---|---|---|---|---|---|---|---|---|
| A1 | `stack/tanitad/resim/export.py` | 454 L | **2026-08-13** | Session-bundle exporter: `TimestepRecord`s → `session.json` + `frames/*.jpg`. Owns the bundle schema, branded palette, `uncalibrated_corpora` fallback, `portable_basename` | library | `replay_app.py:166`, `resim_app.py:36`, `sample.py`, tests | CURRENT | ⭐ **KEEP-AS-CORE** — the P9 data contract |
| A2 | `stack/tanitad/resim/static/app.js` | 1220 L | 2026-07-25 | The SPA: home cards, per-arm columns, camera canvas + fan, decoded-intent HUD, shared BEV master panel, error strip, scrubber, head readouts | web app | served by A6 | CURRENT | ⭐ **KEEP-AS-CORE** — the UI |
| A3 | `stack/tanitad/resim/static/style.css` | 298 L | 2026-07-25 | Design language (dark slate; MAIN gold, REF-A cyan, REF-B magenta, GT white-dashed) | asset | A4 | CURRENT | KEEP-AS-CORE |
| A4 | `stack/tanitad/resim/static/index.html` | 26 L | 2026-07-25 | SPA shell | asset | A6 | CURRENT | KEEP-AS-CORE |
| A5 | `stack/tanitad/resim/sample.py` | 243 L | 2026-07-25 | Deterministic synthetic bundle exercising every element incl. one fallback episode | CLI + lib | `resim_app.py:136`, tests | CURRENT | **KEEP-AS-CORE** — the contract's executable spec |
| A6 | `stack/scripts/resim_app.py` | 151 L | 2026-07-25 | FastAPI single-port server; `build_app()` + CLI; 5 routes; path-traversal guarded | CLI | tests | CURRENT | ⭐ **KEEP-AS-CORE** → `tanitresim serve` |
| A7 | `stack/tanitad/replay/engine.py` | 450 L | — | Replay engine: `TimestepRecord`, `ArmOutput`, `WAYPOINT_STEPS` | library | A8, A1, A11 | CURRENT | KEEP — the P7/P9 seam (§5.1) |
| A8 | `stack/tanitad/replay/arms.py` | 572 L | **2026-08-14** | Per-arm adapters producing `ArmOutput` | library | A7, A11, `compare_arms.py:753` | **CURRENT, INCOMPLETE** | **MERGE-INTO** — must be extended (§3.1, §3.2) |
| A9 | `stack/tanitad/replay/rr_log.py` | 417 L | 2026-07-12 | **Two things**: the rerun entity schema *and* the camera projection the bundle uses (`to_image_plane`, `cam_for_corpus`) | library | A1, A11, `pod_ops/horizon_probe.py:18` | Mixed | **SPLIT** — projection → shared core; rerun schema → DEPRECATE (§9 Q4) |
| A10 | `stack/tanitad/replay/stats.py` | 238 L | 2026-07-11 | `--mode test` metrics (ADE/FDE, action MAE, manoeuvre accuracy). No graphics | library | A11 | CURRENT | **DEPRECATE** — metrics are P7's (§9 Q3) |
| A11 | `stack/scripts/replay_app.py` | 450 L, ~40 flags | 2026-07-20 | `--mode test\|viz\|export`. Only `export` is P9 | CLI | — | **OVERLOADED** | **SPLIT**: `export` → `tanitresim export`; `viz` → DEPRECATE; `test` → P7 |
| A12 | `stack/tests/test_resim.py` | 33 `def test_` + 2 parametrised (**39 collected** per NOTE.md) | — | Schema, portability, projection-below-horizon, nav_commands, fallback, FastAPI TestClient | tests | — | CURRENT | **KEEP + EXTEND** (§6.2) |
| A13 | `stack/tests/test_replay.py` | 18.3 KB | — | Engine/arms tests | tests | — | CURRENT | KEEP |

### 2.2 Family B — the batch/offline renderers (`taniteval/`)

| # | Path | Lines | Last commit | What it renders | Output | Referenced by | Status | **Disposition** |
|---|---|---|---|---|---|---|---|---|
| B1 | `taniteval/taniteval/corpus_overlay.py` | 463 | 2026-07-27 | "THE STANDARD". Camera (flat pinhole **or** per-clip exact cosmos calib) + 152×196 BEV inset + 3-line HUD; grounded operative rollout | MP4 / PNG | B2, B3, B4, B5, B6, `test_ego_guard.py:273` | **CURRENT — most-imported node** | ⭐ **KEEP-AS-CORE (batch)**; extract primitives |
| B2 | `taniteval/tools/render_openloop_video.py` | 407 | **2026-08-05** | Same three panels, **large 420 px BEV**, scrolling ADE trace, anti-over-claim banner, provenance sidecar | MP4 + `.mp4.json` | `test_render_openloop_video.py` | **CURRENT — best-engineered** | ⭐ **KEEP-AS-CORE (batch)** — its *packaging* is the CLI model (§5.0) |
| B3 | `taniteval/taniteval/direct_overlay.py` | 269 | 2026-07-27 | Same picture for **REF-B / REF-C** (direct heads, no `step_readout`). Adds `_fit()` HUD guard + `densify()` | MP4 / PNG | B4, B5 | CURRENT | **MERGE-INTO B1** as an *arm adapter*, not a second renderer |
| B4 | `taniteval/taniteval/plan_fan.py` | 701 | 2026-07-27 | REF-C **plan fan**: 1280×800 composed panel, 860×692 BEV star + 356 px camera + score colorbar + legend; oracle-in-fan diagnostic | MP4 / PNG | B5 | **CURRENT — most sophisticated** | **KEEP** as the *fan view*; re-base on the core |
| B5 | `taniteval/taniteval/plan_fan_clips.py` | 397 | 2026-07-27 | Six-category window selection + both-arms-same-window driver for B4 | MP4 set + JSON | — | CURRENT | **KEEP** — its selector is what the UI needs too (§8) |
| B6 | `taniteval/taniteval/label_overlay.py` | 352 | 2026-07-25 | Label audit: 4 label sources (kin v2 / v2.1 / VLM / model) colour-coded by disagreement; 176 px HUD | MP4 + JSON | `test_val_parity.py:342` | CURRENT (special-purpose) | **KEEP, RE-BASE** — a distinct view, not a duplicate |
| B7 | `taniteval/taniteval/flagship_overlay.py` | 204 | 2026-07-27 | Camera + ADE only. **Superseded as a CLI** — but is the *style-constants module* (`K`, `WINDOW`, `WP_IDX`, `S`, `COL_*`, `HUD_*`, `_font`) | MP4 | B1–B6, B2, `test_render_openloop_video.py:109` | **live as a library, dead as a CLI** | **SPLIT**: constants → core; `main()` → DELETE |
| B8 | `taniteval/taniteval/cam_overlay.py` | 143 | 2026-07-27 | REF-B-only overlay. **Superseded as a CLI** — but is the *projection library* (`project`, `ego_future_path`, `F_EFF`, `CAM_H`, `UP`) | MP4 | B1–B6, `cosmos_fit`, `cosmos_scan`, `cosmos_gate_stills`, `probe_overlay` | **live as a library, dead as a CLI** | **SPLIT**: projection → core; `main()` → DELETE |
| B9 | `taniteval/taniteval/viz.py` | 70 | 2026-07-20 | Self-contained **SVG** BEV gallery (best/median/worst by ADE), zero deps | SVG strings | *(no importer found)* | **ORPHANED** | **KEEP, RE-PURPOSE** — the only dependency-free renderer; ideal for report embedding |
| B10 | `taniteval/probe_overlay.py` | 61 | 2026-07-27 | ⚠️ **Not a renderer.** Console probe; prints route/manoeuvre/ADE. Hardcoded pod paths; **executes at import** (no `main()`) | stdout | pinned in `test_ego_guard.py:274` | ABANDONED as viz | **DEPRECATE** — move under `taniteval/tests/`; keep the ego-guard pin |
| B11 | `taniteval/cosmos_gate_stills.py` · `cosmos_calib_probe.py` · `cosmos_fit.py` | 137 / 82 / 62 | — | Calibration-fit stills (4 colour-coded candidate projections) | PNG | — | CURRENT (tooling) | **KEEP** — they produce the `calibration-probes` campaign; fold into `tanitresim calib` |

### 2.3 Family C — the simulation renderers (`stack/experiments/alpasim-gsplat/`)

| # | Path | Lines | What it renders | Output | Status | **Disposition** |
|---|---|---|---|---|---|---|
| C1 | `overlay_video.py` | 362 | ⭐ Closed- **or** open-loop: camera (**true f-theta**, `tanitad.data.calib.ftheta_project_ray`) + metric BEV + HUD carrying planned/executed/logged manoeuvre, **route head vs route logged separately**, cross-track, and **lateral + longitudinal metric families**. `cv2.VideoWriter` because Thor has no ffmpeg | MP4 | **CURRENT — highest compliance in the programme** | ⭐ **KEEP-AS-CORE (sim)**; its HUD is the model for §6 |
| C2 | `gsplat_renderer.py` | 1011 | The NuRec gaussian-splat **sensor-sim render core** (11 importers) | RGB arrays | CURRENT | **OUT OF P9 SCOPE** — it renders the *world*, not the *evaluation*. P9 consumes its frames |
| C3 | `render_quality.py` · `render_diagnose.py` · `make_before_after.py` · `frame_align.py` | 637 / 249 / 116 / 478 | Render-fidelity harness, artifact localisation, before/after sheets, frame-offset adjudication | PNG / JSON | CURRENT | **OUT OF P9 SCOPE** — sim-fidelity QA, not eval viz |
| C4 | `closedloop_drive.py` · `openloop_drive.py` · `sensorsim_gsplat_server.py` | 712 / 328 / 279 | Drive loops dumping rendered frames; gRPC sensor service | JPG | CURRENT | OUT OF SCOPE — C1 consumes their output |

⚠️ **Family C is the renderer that actually produced the `alpasim-closedloop-*` and
`alpasim-openloop-*` campaigns** in §4 — i.e. the programme's only closed-loop videos
came from the family the charter does not mention.

### 2.4 Adjacent and deliberately out of scope

| Path | Lines | Why it is not P9 |
|---|---|---|
| `stack/scripts/ph0_rich_overlay.py` · `ph0_overlay_video.py` · `ph0_v2_overlay.py` · `ph0_ab_video.py` · `ph0_sam3.py` | 524 / 437 / 333 / 189 / 1567 | **Perception** overlays (SAM3 masks, sign boxes, OCR). A different subject: what the *perception* pipeline saw, not what the *planner* decided. Shares the compositing convention; should share the core (§7 step 3) but stays its own view |
| `stack/scripts/p8_bev_reel.py` | 406 | **World-model belief** view: camera │ decoded belief BEV │ GT BEV raster, scored by occupancy IoU. Distinct concern; no trajectory HUD |
| `stack/tanitad/data/bev_raster.py` · `taniteval/taniteval/lane_raster.py` | 541 / 147 | **Model inputs**, not visualisations. Natural sources of a future map/occupancy pane (§9 Q5) |
| `Paper/figures/make_lf0_bev_panels.py` · `make_v58f_results.py` · `make_winners_curse.py` | 272 / 228 / 388 | Hand-rolled SVG **paper figures**. P10 territory |
| `stack/scripts/eval_behavior.py` · `validate_refb_labels.py` · `validate_geometry.py` · `probe_saliency_p9.py` · `geom_sanity.py` | 947 / 603 / 553 / 625 / 365 | Diagnostic plots inside analysis tools. Would benefit from the shared palette; not renderers |
| `stack/scripts/visualize_episode.py` · `viz_trajectory_fan.py` | 146 / 222 | Early prototypes, superseded by B1/B4. **DELETE candidates** — no importer found |
| `colab/s2_lab_lib.py` + 2 notebooks · `DataEng/.../Loading_NvidiaDataSet.ipynb` | 1182 / 547 | Notebook/lab tooling |

### 2.5 The long tail — ~25 banked one-off renderers

Under `TanitAD Research Lab/**/incoming/**/` and `stack/experiments/{reset-speed4b,pod-rescue-*,nurec-gsplat}/`.
Representative: `render_v5f_bev.py` (178 — camera + BEV fan + HUD + **tactical/strategic
goal strip** + sel ADE, and the compositing convention `ph0_overlay_video.py` follows),
`render_v5f_fanfull.py` (234), `render_v16.py` / `render_v16_cont.py` /
`render_v17_cont.py` (116/90/90), `a2_compare_video.py` (212), `idmval_render.py` (245),
`reset-speed4b/{render_arm.py, build_svg_gallery.py, build_ascii.py}` (330/151/51),
`make_v2_videos.py` (28), the SAM3/tactical HTML review sheets (718/625/821/…).

**Disposition: FREEZE, do not consolidate, do not delete.** Each is the executable
provenance of a banked result and re-rendering is not free. They are the *symptom* the
consolidation treats: when the shared CLI can do the job, new one-offs stop appearing.
The one action is a **generated index** listing which campaign each produced (part of
§7 step 1).

### 2.6 Excluded after checking (named so nobody re-checks them)

`render_tables.py` (×4), `render_gate.py`, `render_sweep_table.py`, `rebaseline_table.py`,
`p8_render.py` — all emit **markdown tables**, zero graphics calls. `stack/tanitad/replay/stats.py`
is aggregation only. `tools/registry_paths.py`, `tools/secret_scan.py` merely match
`*.mp4` as filename patterns. `stack/scripts/ph0_pilot.py`, `ph0_v2.py`, `d8_preview.py`
decode MP4 but draw nothing.
**Confirmed empty:** `stack/tools/` does not exist; repo-root `tools/` has no graphics;
`_pod_backup/` holds only a status file, a diff and a ckpt log; the twelve
`stack/experiments/p0-*` dirs (incl. `p0-fan-viz`) contain **no code** despite the names.

### 2.7 ⛔ The dependency chain nobody can delete around

```
cam_overlay        (project, ego_future_path, F_EFF, CAM_H, UP)
  └─ flagship_overlay  (K, WINDOW, WP_IDX, S, COL_*, HUD_*, _font)
       └─ corpus_overlay  (FlatProjector, ExactProjector, draw_frame, draw_bev,
                           clip_extent, pretty_man, pretty_route, HORIZON)
            ├─ direct_overlay  (OUT, WP_STEPS, _fit, densify)
            │    └─ plan_fan  →  plan_fan_clips
            ├─ label_overlay
            └─ tools/render_openloop_video
```

**`cam_overlay.py` and `flagship_overlay.py` look like dead superseded CLIs and are
not.** Every other batch renderer imports its geometry and palette from them; deleting
either breaks six modules and two test files. Their `main()`s *are* dead. This is the
most likely way to break P9 while "cleaning it up".

---

## 3. Viz-standard compliance matrix

The standard (Sayed, standing; restated `taniteval/README.md:108-113`): **camera
projection + metric BEV + decoded tactical manoeuvre + strategic route/goal, together,
with ADE**; BEV-only fallback when camera calibration is unrecoverable.

Verified by **reading the drawing code**, not docstrings. `✅` drawn and correctly
sourced · `⚠️` drawn but the source is wrong · `❌` absent.

| Implementation | 1 Camera | 2 Metric BEV | 3 Tactical | 4 Strategic | ADE | BEV-only fallback |
|---|---|---|---|---|---|---|
| ⭐ **`overlay_video`** (C1) | ✅ **true f-theta** `:18`, not a pinhole approximation | ✅ `draw_bev:96-142`, ego-centred, range rings | ✅ planned / executed / logged manoeuvre `:24` | ✅ **`route head:` AND `route logged:` as separate fields** `:209-211` | ✅ + **lat/lon families** `:214-218` | ❌ n/a (sim calib always known) |
| **`corpus_overlay`** (B1) | ✅ `draw_frame:231-245`, flat **or** exact per-clip | ✅ `draw_bev:186-223` (152×196 inset) | ✅ `tactical_policy.maneuver_logits.argmax` `:319` → `:363` | ✅ `strategic_policy.route_logits.argmax` `:317` → `:363` | ✅ `:364` | ✅ **the only one in Family B** — `:246-248` draws *"camera overlay disabled — calibration unverified (see BEV)"* |
| **`render_openloop_video`** (B2) | ✅ `:295-304` | ✅ **large 420 px** `:314` + ADE trace `:317` | ✅ `co.pretty_man` `:340` | ✅ `co.pretty_route` `:341` | ✅ `:344-346` + rolling + trace | ❌ **unreachable** — `proj = co.FlatProjector(128.0)` `:254` hardcoded |
| **`direct_overlay`** (B3) | ✅ imports `draw_frame` | ✅ via `draw_frame` | ✅ `o["maneuver_logits"].argmax` `:140` | ✅ `o["route_logits"].argmax` `:141` | ✅ `:176` | ❌ **unreachable** — `FlatProjector(cy)` `:220` never `None` |
| **`plan_fan`** (B4) | ✅ 356 px context panel | ✅ 860×692, the star | ✅ `pretty_man` | ✅ `pretty_route` | ✅ sel / oracle / vocab | ❌ `FlatProjector` only |
| **`label_overlay`** (B6) | ✅ `:118-130` | ✅ `draw_bev` `:131` | ✅ kin-v2 label **+** model argmax `:238-240` | ✅ 4 route rows, colour-coded `:241-256` | ⚠️ only with `--model` | ❌ `FlatProjector(args.horizon)` `:313` |
| `render_v5f_bev` (§2.5) | ✅ | ✅ fan pane | ✅ `:136,154` | ✅ `:154` | ✅ `:94` | ❌ |
| **TanitResim SPA** (A2) | ✅ per-arm camera canvas + fan | ✅ shared BEV master, metre grid + scale bar + legend | ✅ `maneuver_probs` argmax | ⚠️ **§3.1 — this is not a strategic output** | ✅ HUD + header + strip | ✅ `uncalibrated_corpora` → `wp_img=null` (`export.py:324,349`) |
| `flagship_overlay` (B7) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `cam_overlay` (B8) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `viz.py` (B9) | ❌ | ✅ SVG | ❌ | ❌ | ✅ in title | — |
| `p8_bev_reel` (§2.4) | ✅ | ✅ belief + truth | ❌ | ❌ | ❌ (occupancy IoU) | ❌ |

**Reading of the matrix:** the standard is **met by five implementations and by none of
them completely**. `corpus_overlay` is the only one with a working fallback;
`overlay_video` is the only one with a correct camera model and the only one that
separates a predicted route from a logged one; `render_openloop_video` is the only one
with portable packaging. **No single file is the merge target — the core has to be
assembled from three.**

### 3.1 ⛔ FINDING — the SPA's "strategic" element renders a ground-truth **input**

`stack/tanitad/resim/README.md:80` maps standard row 3b — *"Strategic route/goal text
(`route_logits` argmax)"* — onto the SPA's HUD *"from `nav_cmd` via `meta.nav_commands`"*.
Those are different quantities, and the code says so:

- `stack/tanitad/replay/engine.py:220` — `nav_cmd: int | None = None   # strategic input (REF-B)`
- `stack/tanitad/replay/arms.py:476-479` — *"The strategic nav command is DERIVED per
  window from route-scale future heading (`scripts/refb_labels.nav_command`) — **it is
  the navigator's input to the model, not a prediction**"*
- `stack/tanitad/replay/arms.py:526` — `nav = [self._labels.nav_command(ref.episode.poses, ref.last)[0] …]`
  — computed from **the episode's own future poses**.

So the SPA's `strategic: route <goal>` is (a) not a model output, and (b) derived from
the ego's own future path — the construction CLAUDE.md binds against: *"A supplied route
is optimistic by construction on PhysicalAI — our only route supplier there is the ego's
own future path"*, and the nav-echo defect where a route head that is *"an exact
bijection of the nav we feed it … scored 1.0000"* was read as skill.
`corpus_overlay:317` reads `route_logits.argmax` and is correct. **`overlay_video.py:209-211`
already does the right thing and is the template: it prints `route head:` and
`route logged:` as two separate fields.**

**Consequence: TanitResim's UI is 3/4 compliant, not 4/4, and its README asserts
otherwise.** The pixels are fine; the caption is wrong. The green test suite missed it
because `test_meta_carries_nav_commands` (`test_resim.py:267-275`) asserts only that the
value is *present and correctly labelled from* `meta.nav_commands` — never *what it
means*. **Until fixed, no screenshot of the SPA's HUD may be quoted as
evidence about strategic decision quality.** ⭐ Escalated.

### 3.2 The second gap (recorded, re-verified as a mechanism)

`…/2026-07-24-tanitresim-productionization/NOTE.md:108-114` records that the flagship
(`main`) arm does not emit `maneuver_probs`/`nav_cmd` in `arms.py` — only REF-B does —
so on a **real** bundle the main arm's HUD shows only `ADE · v`. I re-verified the
mechanism (`arms.py:558-565` populates those fields only in the REF-B adapter) but did
**not** run a live bundle: *that the flagship arm is blank in practice* is **INHERITED**.

**Net:** the UI's tactical and strategic elements are today populated **only for REF-B**,
and the strategic one is an input. Families B and C are the only places elements 3 and 4
are currently correct for the arms we ship.

---

## 4. Existing media

Verified by content — sizes, MP4 box headers, md5s — never by filename or count.

| Location | Campaigns | Files | Size | Notes |
|---|---|---|---|---|
| **Worktree** `TanitAD Research Lab/Evaluation/Videos/` | 17 | 200 | 486.4 MB | png 150 · **mp4 28** · md 12 · json 7 · jpg 2 · log 1 |
| **Main repo** `TanitAD Research Lab/Benchmarks & Evals/_evaluation/Videos/` | 20 | 268 | 558.9 MB | the full set — **Drive-dehydrated, see below** |
| **Backup** `C:\Users\Admin\tanitad-media-backup\Videos\` | 20 | 268 | 558.9 MB | fully hydrated and readable |
| Backup staging dirs | — | 88 | 40.1 MB | `videos-2026-07-20/`, `videos-2026-07-21/` — largely duplicates |
| Backup root | — | 5 | 13.2 MB | `.parquet` selection tables |

**The `Research Hub` → `Research Lab` rename is real but has not reached this worktree
branch, and it was more than a rename** — `Evaluation/` also moved under
`Benchmarks & Evals/_evaluation/`. Both layouts exist right now in different checkouts.

**Media are real.** Zero 0-byte files either side; every sampled MP4 opens with a valid
`ftypisom` box; the smallest MP4 (96,832 B) is listed at that size in the campaign's own
`INDEX.md`. Five md5s recorded in READMEs match the backup byte-for-byte
(`v17_continuous.mp4` `6fb443a9…`, `v5f_bev_compact.mp4` `d9151ad8…`,
`v5f_fan_compact.mp4` `c76f0527…`, `v5f_planfan_compact.mp4` `656cdac7…`,
`v16_vs_v1arch_reel.mp4` `ef65fd27…`).

**Date range: UNVERIFIED from the filesystem.** Every worktree mtime is the checkout
time; every backup mtime is the backup time (2026-08-23 ~00:58). Real render dates live
in folder names and READMEs and span **2026-07-20 → 2026-08-12**.

### 4.1 ⛔ 68 MP4s are stranded on one disk

`.gitignore:24` is `*.mp4` (under `# Data (never commit datasets)`, lines 21-25;
`.png`/`.jpg`/`.json`/`.md` are **not** ignored).

| | count |
|---|---|
| Media files (mp4+png+jpg) in the full set | 248 |
| **Tracked** | **180** — 28 mp4 (force-added past the ignore), 150 png, 2 jpg |
| **Untracked / gitignored** | **68 — all `.mp4`** |

⭐ **Those 68 videos exist in exactly two places: the main-repo working tree and
`C:\Users\Admin\tanitad-media-backup\`, and they are not recoverable from git.** The
main-repo copies are currently **cloud-only Google Drive placeholders** — `Get-FileHash`
fails on every one with *"Unzulässige Funktion"*. **The `C:` backup is therefore the sole
locally-readable copy of 68 rendered campaign videos, on one consumer disk, with no
second copy and no checksum manifest.** Included: all of `REF-A`, `REF-B`,
`flagship-v1` (campaigns that exist *only* in the backup), 18 `planfan-clips`,
12 `v2corpus-vs-v1`, 9 `REF-C`, 14 `other`.

**RECOMMENDATION — for the PI, not for an agent to execute:**

1. **Do not `git add -f` the 68 MP4s.** ~72 MB of already-tracked binaries is enough;
   68 more permanently inflates every clone. *(I have not force-added anything.)*
2. **Do** generate and track `Videos/MEDIA_MANIFEST.json` — path, bytes, md5, campaign,
   render date, producing renderer + commit. It is text, it makes the backup verifiable,
   and it turns "is the backup complete?" into a command. **~1 h, 0 GPU, highest value
   here.**
3. **Then** choose a durable home for the bytes. ⚠️ `plan_fan_clips.py:46-47` binds:
   *"the renders stay on the pod and in the repo. Never push frames, crops or renders to
   HF or any external service."* **That constraint and the need to get off one disk are
   in direct tension and only the PI can resolve it** (§9 Q1).
4. `INDEX.md` is **stale** — it does not mention `alpamayo-showcase-2026-08-07`,
   `alpamayo2-vs-flagship-2026-08-06`, or `ph0-vlm-overlay-2026-08-12`. The manifest in
   (2) should be generated, not hand-maintained.

---

## 5. CLI mode spec

One console entry, `tanitresim`, from `stack/pyproject.toml`. It replaces five CLIs
(`corpus_overlay`, `direct_overlay`, `flagship_overlay`, `cam_overlay`,
`render_openloop_video`) and absorbs one mode of `replay_app.py`.

### 5.0 Global rules — each earned by a defect above

- **No hardcoded absolute paths.** Six of the eight Family-B renderers hardcode
  `OUT = Path("/root/taniteval/results/videos")` and `device = "cuda"`; they cannot run
  on the dev box or Thor without editing. Every path and the device become flags with
  env fallbacks (`TANITRESIM_OUT`, `TANITRESIM_SESSIONS`).
- **`--out` is a file for one artifact, a directory for a set**; a mismatch exits before
  any frame is rendered (`render_openloop_video:212-213`).
- **The encoder is resolved before the first frame** — PATH `ffmpeg`, then
  `imageio-ffmpeg`, then `cv2.VideoWriter` (`render_openloop_video:60-76`; C1 needs the
  cv2 path because **Thor has no ffmpeg**, `overlay_video.py:27`). Never after: a
  30-episode run discovers a missing encoder several thousand frames too late.
- **Every subcommand that emits pixels emits a sidecar JSON**: arm key + ckpt + step,
  corpus + episode ids, selection method verbatim, the §6 element report, resolution,
  fps, frame count, and the harness artifact ids consumed.
- **Selection provenance is burned into the frame**, not only the sidecar
  (`render_openloop_video:191-204`): `spread` / `first` / `explicit:<label>`.
  **⛔ A bare `best` is not offered** — a hand-picked reel must never be quotable as
  representative.
- **Every HUD string goes through `wrap()`/`fit()`.** `render_openloop_video:79-99`
  records the measured failure: at 962 px the banner clipped at *"NOT autonomous
  driving, N…"*, so *"NOT a hierarchy result"* was absent from **every frame of every
  reel** and looked fine in a downscaled still.

### 5.1 The P7 seam (dependency — do not duplicate)

TanitResim **consumes** the eval harness's artifacts and **never recomputes a scored
quantity**. A sibling agent is auditing `taniteval/`'s data contract; this spec
deliberately does **not** restate that schema. The seam has three abstract parts:

> **`ArmHandle`** — arm key → `{model, step_readout|None, feed, speed_input, dyn_input,
> yaw_input, window, arch, step}`. Today: `taniteval.loaders.load` + `registry.MODELS`.
> **`WindowSource`** — an episode/window iterator **with parity enforcement**. Today:
> `taniteval.data.list_val_episodes` — the val-parity chokepoint, which **must remain
> the only entry** (pinned by `tests/test_val_parity.py:340-342` over every overlay
> module; a bare glob is what that test exists to forbid).
> **`BankedResult`** — per-window dumps + eval JSON, used for *selection* and for the
> ADE burned into the HUD. Today: `taniteval/results/windows_<arm>.pt`, `fan_<arm>.pt`.

**Binding:** the ADE displayed must be **the same quantity the leaderboard row reports** —
which is why `direct_overlay:17-19` calls each arm exactly as its scoring collector
does. Any refactor preserves that. If the P7 audit changes the contract, P9 adapts; P9
does not fork it.

### 5.2 Command surface

```
tanitresim render   batch renderer (MP4 / stills). Replaces 5 CLIs.
tanitresim fan      the plan-fan view (REF-C anchored diffusion; reusable for CEM).
tanitresim labels   the label-audit view (4 label sources, disagreement-coloured).
tanitresim sim      closed-/open-loop AlpaSim reels (wraps C1).
tanitresim calib    camera-calibration fit stills (wraps B11).
tanitresim export   build a portable session bundle for the UI.
tanitresim serve    serve bundles (the web app).
tanitresim demo     synthetic bundle + serve; no pod, no checkpoint.
tanitresim doctor   check viz-standard compliance of a bundle/sidecar. No GPU.
```

#### `tanitresim render`

```
tanitresim render
  --arm KEY[,KEY...]          registered arm key(s); >1 → one reel per arm
  --ckpt PATH [--run-config PATH]     override the registry entry
  --corpus KEY|PATH           registered corpus key, or an ep_*.pt directory
  --out PATH                  .mp4 file (single) | directory (set)
  --select spread|first|explicit          [spread]
  --episode-list 3,17,28      with --select explicit
  --select-label STR          what the list means, e.g. "worst-30-by-ADE"; burned in
  --episodes N                                          [30]
  --max-frames-per-ep N       0 = all                    [0]
  --stills [N] | --thumbs     PNG instead of / alongside a video
  --bev large|inset|only      pane size; `only` forces the fallback layout   [inset]
  --fps N [10]   --device auto|cuda|cpu [auto]
  --calib auto|flat|exact|ftheta|none     camera model
  --horizon-row FLOAT         override the flat-model horizon (flat corpora only)
  --require-all-elements      exit non-zero if any of the 4 is unavailable  [off]
```

**Arm dispatch is automatic — this is the B1+B3 merge.** An arm with a `step_readout`
renders via the grounded operative rollout; an arm without one (REF-B/REF-C) renders via
its direct head with `densify()` as a drawing device only. The user does not choose the
branch: choosing it wrong is exactly the error `direct_overlay:230-233` asserts against.

**Input contract:** `--arm` → `ArmHandle`; `--corpus` → `WindowSource`; `--select
explicit` → `BankedResult` for the ranking. Nothing else.
**Output:** `<out>.mp4` + `<out>.mp4.json`, or `<out>/*.png` + `<out>/manifest.json`.

#### `tanitresim export`

```
tanitresim export
  --arms NAME:CKPT[:VARIANT] ...      NAME drives the branded colour
  --corpus KEY|PATH  --corpus-glob GLOB
  --episodes N  --stride N  --batch N
  --out DIR  --session-name STR
  --uncalibrated-corpora KEY[,KEY]    force the BEV-only fallback
  --uncalibrated-episodes EP[,EP]     ⭐ NEW — per-clip grain (§6.3)
  --jpeg-quality N [80]  --max-width N [640]
  --gates / --no-gates                attach the D1-D3 formal-gate block
  --device auto|cuda|cpu
```

Output: the existing bundle — `session.json` (relative paths only) + `frames/ep<i>_step<j>.jpg`.
**The bundle schema is `export.py`'s and does not change** (it is already portable,
absolute-path-leak tested, and fail-loud on empty/frameless streams); this is
`replay_app.py --mode export` with a smaller, renamed flag set plus the two §6 additions.

#### `tanitresim serve` / `demo`

```
tanitresim serve --sessions-root DIR [--port 8888] [--host 0.0.0.0] [--static DIR]
tanitresim demo  [--port 8888]
```

Unchanged from `resim_app.py` — single plain-HTTP port, which is what survived the
RunPod proxy where the rerun gRPC-web stream did not.

#### `tanitresim doctor` — the contract checker

```
tanitresim doctor PATH [--strict] [--media]
```

`PATH` = a bundle dir, a `session.json`, or a render sidecar. Prints the four-element
availability report; `--strict` exits non-zero if any element is **absent without a
declared reason**; `--media` verifies `MEDIA_MANIFEST.json` against a directory.
**No GPU, no model load, no re-render** — metadata only. This is what makes §6
enforceable in CI and in review.

---

## 6. The viz standard as an enforced contract

Today compliance is a **docstring claim and a README table**, and §3.1 is exactly how
that fails: the README asserts an equivalence the code contradicts, and 39 green tests
missed it because none asserts *what a value means*, only that it is present.

The fix is to make the contract a **data structure that travels with the artifact** —
the programme's per-family reporting discipline applied to pixels (*a missing metric is
a work item, not an excuse; where a family cannot be computed, say so per family with
the reason and the n*).

### 6.1 The `VizElement` record

```python
@dataclass(frozen=True)
class VizElement:
    element: Literal["camera", "bev", "tactical", "strategic", "ade"]
    state:   Literal["present", "unavailable"]
    source:  str | None   # ⛔ REQUIRED when present — the exact expression, e.g.
                          #   "strategic_policy.route_logits.argmax(-1)"   model OUTPUT
                          #   "refb_labels.nav_command(poses, last)"       GT-derived INPUT
    kind:    Literal["model_output", "given_input", "gt_label", "derived"] | None
    reason:  str | None   # ⛔ REQUIRED when unavailable; drawn on the frame
    n:       int | None   # frames/windows the state applies to
```

Three rules, each earned by a defect above:

1. **`source` is mandatory when `present`.** §3.1 existed because nothing forced the SPA
   to declare *where* its strategic value came from. One declared line makes an
   input-masquerading-as-output visible at a glance.
2. **`kind` is mandatory and is rendered.** An element whose kind is `given_input` or
   `gt_label` draws with a **privileged marker** — `strategic: route left ⓘ given` —
   because a GT-derived route on screen is optimistic by construction and the frame must
   say so. `overlay_video.py:209-211` already ships the two-field version of this and is
   the reference implementation.
3. **`reason` is mandatory when `unavailable`, and it is drawn on the frame**, not only
   in the sidecar. `corpus_overlay:246-248` already does this for the camera; the pattern
   generalises to all four. An arm with no policy brains must render
   `tactical: n/a — arm has no tactical_policy`, never a blank.

### 6.2 Where it is enforced

| Point | What it does |
|---|---|
| `render_artifact()` | Refuses to write an MP4/PNG if any of the five records is missing. A *declared* unavailability is fine; an *undeclared* one is a hard error |
| Sidecar / `session.json` | `viz_elements: [...]` is a required top-level key |
| Frame compositor | Draws the HUD **from the records**, so an unavailable element renders its reason instead of vanishing |
| `tanitresim doctor --strict` | CI gate and the reviewer's one-liner |
| `pytest` | (a) parametrised over every registered renderer × every element — a record must exist; (b) ⭐ **a pin that `strategic.source` names a model output for any arm claimed to have a strategic brain** — the §3.1 regression test; (c) `sample.py` keeps exercising all five including one `unavailable`; (d) a Playwright smoke over the SPA's four HUD elements + the fallback note (the currently-untested frontend) |

### 6.3 Per-clip, not per-corpus

`uncalibrated_corpora` is per-**corpus** (flagged in NOTE.md:115-117). `corpus_overlay`
is already per-**clip**: `cosmos_projectors()` returns `None` for an episode with no
verified calib entry. The contract adopts the finer grain — `VizElement` is emitted
**per episode**, and per-corpus becomes a convenience that expands to per-episode at
export time.

---

## 7. Consolidation plan

Ordered by value-per-risk. **Each step is independently shippable and leaves the tree
green.** Effort is implementation time for one agent. **All steps are 0 GPU except
step 9.** Nothing here re-renders existing media.

| # | Step | Effort | Ships | Risk |
|---|---|---|---|---|
| **1** | **Bank the media.** Generate + track `Videos/MEDIA_MANIFEST.json` (path, bytes, md5, campaign, date, producing renderer) for all 268 files, from the readable `C:` backup; add the §2.5 one-off→campaign index; add `doctor --media`. **Binaries stay out of git.** | ~1-2 h | The end of the silent-single-disk state; a verifiable backup | None — text only |
| **2** | ⛔ **Fix the strategic defect (§3.1).** In `export.py` split `nav_cmd` (input) from a new `route_pred` (`route_logits.argmax`); emit `kind` for both; correct the README table; SPA renders the privileged marker. Extend `arms.py` so the flagship arm emits `maneuver_probs` + `route_logits` from its trained `tactical_policy`/`strategic_policy` (§3.2). | ~4-6 h | The UI becomes honestly 4/4 for the arms we ship | **Med** — `arms.py` is shared territory (R1); a live-bundle check wants a GPU box |
| **3** | **Extract `tanitad/viz_core/`** — one module owning the projections (`FlatProjector`, `ExactProjector`, `ftheta_project_ray`, `to_image_plane`, `cam_for_corpus`), palette + fonts + HUD constants, `draw_bev`, `draw_frame`, `wrap`/`fit`, `clip_extent`, `pretty_man`/`pretty_route`, and `VizElement`. `cam_overlay`/`flagship_overlay` become **re-export shims, not deletions** (§2.7); their `main()`s go. | ~6-8 h | One geometry, one palette; the PIL / JS / cv2 panels stop drifting | **Med** — six importers + two test files; mechanical but wide |
| **4** | **Land `tanitresim`** with `render`/`export`/`serve`/`demo`/`doctor`. `render` = B2's packaging + B1's calibration + B3's arm dispatch and `_fit` guard. Old modules keep working via deprecation shims for one cycle. | ~8-10 h | A renderer that runs on the dev box and Thor, not only on a pod | Low-Med |
| **5** | **Enforce the contract (§6).** `VizElement` in every renderer; required sidecar/`session.json` key; `doctor --strict`; the four test classes incl. the §3.1 pin and the Playwright smoke. | ~4-6 h | Compliance becomes checkable instead of asserted | Low |
| **6** | **UI: the three missing operator affordances (§8).** Jump-to-worst, manoeuvre-class filter, cross-episode worst-window drawer. Client-side only. | ~5-6 h | The UI answers "where did it fail?" without a scrub | Low — SPA-local |
| **7** | **Re-base the specialist views** — `plan_fan`, `plan_fan_clips`, `label_overlay`, and `overlay_video` (C1) onto `viz_core` + `VizElement`; `plan_fan_clips`'s six-category selector becomes the shared window-selection service the UI also calls. | ~6-8 h | The fan, label and sim views stop being forks | Low-Med |
| **8** | **Retire the dead surface.** Delete `cam_overlay.main`, `flagship_overlay.main`, `visualize_episode.py`, `viz_trajectory_fan.py`; move `probe_overlay.py` under `taniteval/tests/` (keep its ego-guard pin); demote `replay_app --mode viz` (rerun) and `--mode test` (→ P7). Re-purpose `viz.py`'s SVG gallery as the report-embedding renderer. | ~3-4 h | Five fewer entry points; one obvious way to render | Low |
| **9** | *(optional, needs a GPU box)* **One golden re-render** of a canonical clip through the consolidated path, md5-compared against its banked equivalent. | ~2 h + GPU | Proof the refactor is pixel-faithful | Low |

**Total ≈ 39-52 h, of which only step 9 needs a GPU.** Steps 1, 5, 6, 8 are pure 0-GPU
items suitable for a gated turn.

**Parallelisation — three independent file territories:**
`1` (media + docs) ‖ `2` (`resim/` + `replay/arms.py`) ‖ `6` (`static/app.js`) run
concurrently with no overlap. `3 → 4 → 5` is the serial spine. `7`, `8` follow `3`.

---

## 8. UI mode spec

**Host:** the existing FastAPI single-port server is the primary target (pod or dev
box). The private HF Space **`Sayood/TanitAD`** (Gradio, CPU, unmetered) is a viable
second host **for pre-exported bundles only** — it can serve the static SPA plus a
bundle directory with no GPU and needs no Gradio rewrite (mount the existing ASGI app).
⚠️ It is **blocked for PhysicalAI-AV imagery** by `plan_fan_clips.py:46-47`
(internal-dev-only, never to an external service), so it is usable for synthetic/demo
bundles and non-PhysicalAI corpora. **PI decision — §9 Q1.**

**Principle: build the minimum that delivers the standard.** The SPA already has home
scenario cards, per-arm side-by-side columns, camera canvas + fan + decoded-intent HUD,
a shared metric BEV master panel with scale bar and legend, an all-arm error strip with
click-to-seek, head readouts, a manoeuvre band, a scrubber with keyboard nav
(←/→ step, ↑/↓ episode, space play), and URL-hash deep links
(`#/s/<id>/e/<ep>/t/<step>`). **Four of the operator's five needs are already met.** So
this section is mostly *what not to build*.

| Operator need | Status today | Work |
|---|---|---|
| **Scrub an episode** | ✅ Scrubber, play/pause ~10 fps, keyboard nav | **none** |
| **Compare arms side by side** | ✅ One column per arm + shared BEV + shared error strip, branded colours | **none** |
| **See the standard's four elements** | ⚠️ 3/4 — §3.1 | **step 2** |
| **Jump to worst-ADE windows** | ⚠️ Partial. `meta.episodes[].worst_step` / `worst_ade` are exported (`export.py:411-417`) and shown as a **card badge only** (`app.js:326`). Nothing navigates to them | **NEW (a)** |
| **Filter by tactical manoeuvre class** | ❌ Absent. `maneuver_counts` is exported per episode (`export.py:406-410`) and drawn as a ribbon; nothing filters on it. Grepping `app.js` for a filter control finds only JS array `.filter()` calls | **NEW (b)** |

**(a) Worst-window navigation** — the smallest thing that works:
- The card's worst badge becomes a **link** → `#/s/<id>/e/<ep>/t/<worst_step>` (~10 lines).
- A **"worst windows" drawer** on the home view: the top-N `(episode, step, ade, arm,
  manoeuvre)` rows across the session, sorted, each a deep link. Computed browser-side
  from data already in `session.json` — **no exporter change**.
- In the session view, `[` / `]` step to the previous/next local ADE peak on the error
  strip.

**(b) Manoeuvre-class filter** — a chip row above the card grid, one chip per class from
`meta.maneuver_classes` with counts from `maneuver_counts`. Selecting chips filters the
grid and restricts both the worst-window drawer and `[`/`]` peak navigation to steps
whose `maneuver` matches. `maneuver` is already per-step (`export.py:321-333`), so this
is **client-side only** and needs no re-export.

**Two constraints, both from the standard:**
1. An `unavailable` element renders its **reason text in place** — the camera pane
   already does this for the fallback; the HUD must do it for tactical/strategic.
2. An element whose `kind` is `given_input` or `gt_label` renders with the **privileged
   marker** (§6.1 rule 2). ⛔ **A filter or a ranking must never be built on a
   GT-derived field without that marker visible** — that is how a leak becomes a UI
   affordance.

**Explicitly out of scope:** a build step, any CDN or external asset, a JS framework, a
second port, server-side rendering, and login/multi-user. The no-build / no-CDN /
same-origin design is a feature — it is what survived the RunPod proxy. The one accepted
gap is that the frontend has **no headless test**; the Playwright smoke in §6.2 closes it.

---

## 9. Risks and open questions for the PI

**Q1 — Where do the 68 stranded MP4s live?** (blocks §4.1 step 3) They sit on one
consumer disk with no second copy, but `plan_fan_clips.py:46-47` forbids pushing
PhysicalAI renders to HF or any external service. Options: (a) a second physical disk,
(b) a private HF repo with an explicit carve-out, (c) accept single-copy risk and keep
only the checksum manifest, (d) re-render on demand (GPU cost, and the checkpoints must
outlive the video). **Only the PI can weigh (b) against the imagery rule.**
Recommendation: (a) plus the manifest now; decide (b) separately.

**Q2 — Is §3.1 a defect or a design choice?** This spec assumes a defect (the README
claims `route_logits`, the code ships `nav_cmd`). If the intent was "show the
navigator's instruction", the element is correctly *implemented* and only the *label*
and the compliance table are wrong — a much cheaper fix. Either way the frame must
distinguish an input from a prediction. **Confirm which.**

**Q3 — Does P9 own metrics?** `replay/stats.py` computes ADE/FDE/manoeuvre accuracy
inside the replay path, in parallel with P7's harness — two implementations of the same
quantity, the exact class of drift the programme keeps paying for. This spec assumes
**P7 owns every number, P9 displays it.** Confirm, so step 8 can retire `stats.py`
rather than maintain it.

**Q4 — Rerun: retire or keep?** `replay_app --mode viz` streams to rerun and
`rr_log.py` (417 L) exists for it, but the whole SPA was commissioned *because* the
rerun viewer's legends and signal separation were inadequate and its gRPC-web stream
failed the RunPod proxy. Keeping both doubles the projection surface. This spec proposes
retiring the rerun path and keeping only its projection maths. **Confirm.**

**Q5 — Should a map/occupancy pane be in scope?** `lane_raster.py` (ego lane graph from
NuRec `map.xodr`) and `bev_raster.py` (occupancy from `obstacle.offline`) both exist. A
fifth pane would make strategic decisions *visible* rather than a text label. Genuinely
new scope, not required by the standard. **Defer or schedule?**

**Q6 — Does `overlay_video.py` (C1) become the reference renderer?** It has the best
camera model (true f-theta, not a pinhole approximation), the only correct
predicted-vs-logged route display, and already reports the lateral/longitudinal metric
families in-frame. Promoting it would mean the *sim* renderer sets the standard the
*corpus* renderers follow — sensible on the evidence, but it inverts the current naming
("THE STANDARD" is `corpus_overlay`). **Confirm the direction before step 3.**

**R1 — `arms.py` is shared territory.** Step 2 edits it. Uncoordinated concurrent edits
there are how the index-sweep incidents happened. Sequence it or fence it to one agent.

**R2 — The `Research Hub`/`Research Lab` split is live**, and `Evaluation/` also moved
under `Benchmarks & Evals/_evaluation/`. Two layouts exist simultaneously in two
checkouts. **P9 code must take media paths as arguments, never as constants** — any
constant landed before this settles is wrong in one of them.

**R3 — This worktree's git and filesystem are unreliable.** Google Drive intermittently
fails reads of `.git` internals and of source files (`Invalid request code` /
`Unzulässige Funktion` / `EISDIR` on a plain `.py`). Sources for this spec were pulled
via `robocopy` to local disk and via `git show`, with retries. **Two consequences,
stated rather than hidden:** (i) `stack/tests/test_resim.py`'s last-commit date could
not be obtained, and **no test suite was executed for this spec** — the 33/39 counts are
read from the file and from NOTE.md, so "39 green" is **INHERITED**; (ii)
`taniteval/{conftest,cosmos_calib_build,cosmos_calib_probe}.py` could not be copied
locally after ~20 retries, so their contents are **UNVERIFIED** here.
